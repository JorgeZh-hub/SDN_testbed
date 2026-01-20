# te_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set, Callable, Iterable

from .utils import now

# TE modes (renamed)
MODE_TE_QOS_C = "TE_QOS_C"  # original logic (reactive on congestion)
MODE_TE_QOS_D = "TE_QOS_D"  # lock-driven logic with queue-group dedication
_LEGACY_MODE_MAP = {
    "CONDITIONAL": MODE_TE_QOS_C,
    "AGGRESSIVE": MODE_TE_QOS_D,
}


@dataclass
class _LockEntry:
    group: str
    created_at: float
    expires_at: float


class TEEngine:
    """
    TEEngine supports two modes:
      - te_mode="TE_QOS_C": original logic (only reacts to congestion and checks cooling/MLU/cooldown).
      - te_mode="TE_QOS_D": logic based on link locks + queue groups to attempt per-group dedication.

    Notes:
      - Locks are per directed edge (u,v). If lock_bidirectional=True, it also applies to (v,u).
      - In TE_QOS_D, packet_in handling tries to respect locks; if no "clean" alternative exists,
        it can invade a lock of a lower-priority group (ultimatum) to avoid blocking traffic.
      - In TE_QOS_D, reroutes (run_once) DO NOT invade locks of other groups: if there is no clean path,
        it does not reroute and does not create a new lock.
    """

    def __init__(self,
                 topo,
                 stats,
                 path_engine,
                 flow_mgr,
                 logger,
                 te_period: float = 5.0,
                 cooldown_s: float = 45.0,
                 hot_th: float = 0.85,
                 delta_hot: float = 0.05,
                 delta_global: float = 0.02,
                 Kc: float = 10.0,
                 Ku: float = 5.0,
                 default_link_capacity: float = 100.0,
                 safety_factor: float = 1.2,
                 r_min_mbps: float = 0.1,
                 table_id: int = 0,
                 managed_classes: Optional[List[str]] = None,
                 protect_queues: Optional[List[int]] = None,
                 lower_queue_is_higher_priority: bool = True,
                 unknown_queue_behavior: str = "protect",
                 log_enabled: bool = True,
                 # ---- New (TE_QOS_D mode) ----
                 te_mode: str = MODE_TE_QOS_C,
                 queue_groups: Optional[List[dict]] = None,
                 lock_ttl_s: float = 300.0,
                 lock_bidirectional: bool = True,
                 max_moves_per_run: int = 50):
        self.topo = topo
        self.stats = stats
        self.path_engine = path_engine
        self.flow_mgr = flow_mgr
        self.log = logger
        self.log_enabled = bool(log_enabled)
        self.table_id = int(table_id)

        self.te_period = float(te_period)
        self.cooldown_s = float(cooldown_s)
        self.hot_th = float(hot_th)
        self.delta_hot = float(delta_hot)
        self.delta_global = float(delta_global)
        self.Ku = float(Ku)
        self.Kc = float(Kc)
        self.default_link_capacity = float(default_link_capacity)
        self.safety_factor = float(safety_factor)
        self.r_min_mbps = float(r_min_mbps)

        # Optional class filter: if managed_classes is None/empty => consider all.
        if managed_classes is None:
            self.managed_classes = None
        else:
            mc = {str(x).strip().upper() for x in managed_classes if str(x).strip()}
            self.managed_classes = mc if mc else None

        # Queue protection (applies only to TE_QOS_C). Default: protect highest-priority queue 0.
        if protect_queues is None:
            self.protect_queues = {0}
        else:
            self.protect_queues = {int(x) for x in protect_queues}

        # Priority convention:
        # - If lower_queue_is_higher_priority=True (default): lower queue_id => higher priority.
        self.lower_queue_is_higher_priority = bool(lower_queue_is_higher_priority)

        # If a cookie has no known queue (None):
        # - "protect" (default): do not move it
        # - "lowest": treat it as lowest priority (move it first)
        self.unknown_queue_behavior = str(unknown_queue_behavior or "protect").strip().lower()
        if self.unknown_queue_behavior not in ("protect", "lowest"):
            self.unknown_queue_behavior = "protect"

        # ---- Mode ----
        raw_tm = str(te_mode or MODE_TE_QOS_C).strip()
        tm = _LEGACY_MODE_MAP.get(raw_tm.upper(), raw_tm.upper())
        if tm not in (MODE_TE_QOS_C, MODE_TE_QOS_D):
            tm = MODE_TE_QOS_C
        self.te_mode = tm

        # ---- TE_QOS_D: lock state + queue groups ----
        self.lock_ttl_s = float(lock_ttl_s)
        self.lock_bidirectional = bool(lock_bidirectional)
        self.max_moves_per_run = int(max_moves_per_run) if int(max_moves_per_run) > 0 else 0

        # e=(u,v) -> LockEntry
        self.link_lock: Dict[Tuple[int, int], _LockEntry] = {}

        self._queue_to_group: Dict[int, str] = {}
        self._group_to_queues: Dict[str, Set[int]] = {}
        self._group_rank: Dict[str, int] = {}  # 0 = highest priority

        self._init_queue_groups(queue_groups)

    # ---------------------------------------------------------------------
    # Group configuration
    # ---------------------------------------------------------------------
    def _init_queue_groups(self, queue_groups: Optional[List[dict]]):
        """
        queue_groups (YAML):
          te:
            TE_QOS_D:
              queue_groups:
                - name: GOLD
                  queues: [3]
                - name: SILVER
                  queues: [2,1]
        """
        groups: List[Tuple[str, List[int]]] = []

        if isinstance(queue_groups, list) and queue_groups:
            for g in queue_groups:
                if not isinstance(g, dict):
                    continue
                name = str(g.get("name", "")).strip().upper()
                if not name:
                    continue
                qs = g.get("queues", g.get("queue_ids", []))
                if not isinstance(qs, list):
                    continue
                qids = []
                for x in qs:
                    try:
                        qids.append(int(x))
                    except Exception:
                        continue
                qids = sorted(set(qids))
                if not qids:
                    continue
                groups.append((name, qids))

        # Fallback: one group per queue_id observed in the QoS mapping
        if not groups:
            observed: Set[int] = set()
            try:
                observed |= set(int(v) for v in (self.flow_mgr.class_to_queue or {}).values())
            except Exception:
                pass
            if self.flow_mgr.default_queue_id is not None:
                observed.add(int(self.flow_mgr.default_queue_id))
            # legacy fallbacks
            observed.add(int(getattr(self.flow_mgr, "queue_id_be", 0)))
            observed.add(int(getattr(self.flow_mgr, "queue_id_crit", 1)))
            for qid in sorted(observed):
                groups.append((f"Q{qid}", [qid]))

        # Build maps and validate uniqueness
        q_to_g: Dict[int, str] = {}
        g_to_q: Dict[str, Set[int]] = {}
        for name, qids in groups:
            if name not in g_to_q:
                g_to_q[name] = set()
            for qid in qids:
                if qid in q_to_g and q_to_g[qid] != name:
                    # if duplicated, keep the first and warn
                    if self.log_enabled:
                        self.log.warning(
                            "[TE][AGGR] queue_id=%s aparece en múltiples grupos (%s y %s). Se usará %s.",
                            qid, q_to_g[qid], name, q_to_g[qid]
                        )
                    continue
                q_to_g[qid] = name
                g_to_q[name].add(qid)

        self._queue_to_group = q_to_g
        self._group_to_queues = g_to_q

        # Rank groups by priority derived from their queue_ids
        def prio_key(item: Tuple[str, Set[int]]) -> Tuple[float, str]:
            g, qs = item
            if not qs:
                return (float("inf"), g)
            if self.lower_queue_is_higher_priority:
                # lower qid => higher priority => smaller key
                return (min(qs), g)
            # higher qid => higher priority => smaller key via negative
            return (-max(qs), g)

        ordered = sorted(g_to_q.items(), key=prio_key)
        self._group_rank = {g: idx for idx, (g, _) in enumerate(ordered)}

        if self.log_enabled:
            self.log.info("[TE][AGGR] groups=%s", {g: sorted(list(qs)) for g, qs in self._group_to_queues.items()})
            self.log.info("[TE][AGGR] group_rank=%s", self._group_rank)

    def group_for_queue(self, qid: Optional[int]) -> str:
        if qid is None:
            return "UNKNOWN"
        try:
            q = int(qid)
        except Exception:
            return "UNKNOWN"
        return self._queue_to_group.get(q, f"Q{q}")

    def group_rank(self, group: str) -> int:
        g = str(group or "").strip().upper()
        return int(self._group_rank.get(g, 10**9))

    # ---------------------------------------------------------------------
    # Lock handling
    # ---------------------------------------------------------------------
    def _cleanup_locks(self):
        if not self.link_lock:
            return
        t = now()
        expired = [e for e, ent in self.link_lock.items() if ent.expires_at <= t]
        for e in expired:
            self.link_lock.pop(e, None)

    def _active_lock_group(self, e: Tuple[int, int]) -> Optional[str]:
        ent = self.link_lock.get(e)
        if ent is None:
            return None
        if ent.expires_at <= now():
            return None
        return ent.group

    def active_locked_edges(self) -> Set[Tuple[int, int]]:
        self._cleanup_locks()
        t = now()
        return {e for e, ent in self.link_lock.items() if ent.expires_at > t}

    def locked_edges_for_group(self, group: str) -> Set[Tuple[int, int]]:
        self._cleanup_locks()
        g = str(group or "").strip().upper()
        t = now()
        return {e for e, ent in self.link_lock.items() if ent.expires_at > t and ent.group == g}

    def group_has_home(self, group: str) -> bool:
        return bool(self.locked_edges_for_group(group))

    def set_lock(self, e: Tuple[int, int], group: str):
        if self.lock_ttl_s <= 0:
            return
        g = str(group or "").strip().upper()
        t = now()
        ent = _LockEntry(group=g, created_at=t, expires_at=t + self.lock_ttl_s)
        self.link_lock[e] = ent
        if self.lock_bidirectional:
            rev = (e[1], e[0])
            self.link_lock[rev] = _LockEntry(group=g, created_at=t, expires_at=t + self.lock_ttl_s)
        if self.log_enabled:
            self.log.info("[TE][AGGR] lock_set edge=%s group=%s ttl=%.1fs bidir=%s", e, g, self.lock_ttl_s, self.lock_bidirectional)

    # ---------------------------------------------------------------------
    # Cost helpers
    # ---------------------------------------------------------------------
    def _cost_capacity_only(self, e: Tuple[int, int]) -> float:
        C = self.stats.capacity_mbps(e)
        return (self.Kc * self.default_link_capacity / C) if C > 0 else float("inf")

    def _cost_kc_ku(self, e: Tuple[int, int], sim_load: Optional[Dict[Tuple[int, int], float]] = None) -> float:
        C = self.stats.capacity_mbps(e)
        if C <= 0:
            return float("inf")
        if sim_load is None:
            L = float(self.stats.link_load_mbps.get(e, 0.0))
        else:
            L = float(sim_load.get(e, self.stats.link_load_mbps.get(e, 0.0)))
        U = L / C
        D = self.default_link_capacity / C
        return self.Kc * D + self.Ku * U

    def _path_cost(self, path: List[int], cost_fn: Callable[[Tuple[int, int]], float]) -> float:
        if not path or len(path) == 1:
            return 0.0
        c = 0.0
        for i in range(len(path) - 1):
            c += float(cost_fn((path[i], path[i + 1])))
        return c

    # ---------------------------------------------------------------------
    # Path selection (TE_QOS_D packet_in)
    # ---------------------------------------------------------------------
    def _best_path_via_locked_edge(self,
                                  src: int,
                                  dst: int,
                                  target_edges: Iterable[Tuple[int, int]],
                                  avoid_edges: Set[Tuple[int, int]],
                                  cost_fn: Callable[[Tuple[int, int]], float]) -> Optional[List[int]]:
        best = None
        best_cost = float("inf")
        for (a, b) in target_edges:
            # Validate edge exists in topo
            if self.topo.neigh.get(a, {}).get(b) is None:
                continue
            p1 = self.path_engine.shortest_path(src, a, avoid_edges=avoid_edges, cost_fn=cost_fn)
            if not p1:
                continue
            p2 = self.path_engine.shortest_path(b, dst, avoid_edges=avoid_edges, cost_fn=cost_fn)
            if not p2:
                continue
            path = list(p1) + list(p2)  # introduces (a->b) since p1[-1]=a and p2[0]=b
            # Avoid loops
            if len(set(path)) != len(path):
                continue
            c = self._path_cost(path, cost_fn)
            if c < best_cost:
                best_cost = c
                best = path
        return best

    def pick_path_for_new_flow(self, desc, cookie: int) -> Optional[List[int]]:
        """
        Path selection for packet_in:
          - TE_QOS_C: baseline path (capacity-based).
          - TE_QOS_D: respect locks (prefer home if it exists); if there is no clean path,
            invade lower-priority locks as a last resort.
        """
        if self.te_mode != MODE_TE_QOS_D:
            return self.path_engine.shortest_path(desc.src_dpid, desc.dst_dpid, cost_fn=self._cost_capacity_only)

        self._cleanup_locks()
        sim_load = self.stats.snapshot_loads()
        active = self.active_locked_edges()
        if not active:
            # start with no locks => behave like baseline
            return self.path_engine.shortest_path(desc.src_dpid, desc.dst_dpid, cost_fn=self._cost_capacity_only)

        qid = self.flow_mgr.cookie_queue_id.get(cookie, None)
        group = self.group_for_queue(qid)

        cost_fn = lambda e: self._cost_kc_ku(e, sim_load)

        # 1) Prefer routes that include a group "home lock" (if present)
        forbidden_clean = {e for e in active if self._active_lock_group(e) not in (None, group)}
        home_edges = self.locked_edges_for_group(group)
        if home_edges:
            p = self._best_path_via_locked_edge(desc.src_dpid, desc.dst_dpid, home_edges, forbidden_clean, cost_fn)
            if p:
                return p

        # 2) Clean path (avoid locks from other groups)
        p = self.path_engine.shortest_path(desc.src_dpid, desc.dst_dpid, avoid_edges=forbidden_clean, cost_fn=cost_fn)
        if p:
            return p

        # 3) Ultimatum: allow "invading" locks of lower-priority groups first (avoid higher-priority)
        foreign_groups = {self._active_lock_group(e) for e in active if self._active_lock_group(e) not in (None, group)}
        # Order by lower impact: allow lower-priority groups (higher rank) first
        foreign_ranks = sorted({self.group_rank(g) for g in foreign_groups if g is not None}, reverse=True)
        for thr in foreign_ranks:
            forbidden = set()
            for e in active:
                g = self._active_lock_group(e)
                if g is None or g == group:
                    continue
                if self.group_rank(g) < thr:
                    forbidden.add(e)
            p = self.path_engine.shortest_path(desc.src_dpid, desc.dst_dpid, avoid_edges=forbidden, cost_fn=cost_fn)
            if p:
                return p

        # 4) Final fallback: ignore locks (avoid partition-induced blocking)
        return self.path_engine.shortest_path(desc.src_dpid, desc.dst_dpid, cost_fn=cost_fn)

    # ---------------------------------------------------------------------
    # TE_QOS_D reroute helpers (strict: does NOT invade other groups' locks)
    # ---------------------------------------------------------------------
    def _strict_path(self,
                     src: int,
                     dst: int,
                     group: str,
                     sim_load: Dict[Tuple[int, int], float],
                     avoid_extra: Optional[Set[Tuple[int, int]]] = None,
                     require_home: bool = False) -> Optional[List[int]]:
        self._cleanup_locks()
        active = self.active_locked_edges()
        g = str(group or "").strip().upper()
        avoid = set(avoid_extra or set())
        # Avoid locks from other groups
        avoid |= {e for e in active if self._active_lock_group(e) not in (None, g)}
        cost_fn = lambda e: self._cost_kc_ku(e, sim_load)

        if require_home:
            home_edges = self.locked_edges_for_group(g)
            if not home_edges:
                return None
            return self._best_path_via_locked_edge(src, dst, home_edges, avoid, cost_fn)

        return self.path_engine.shortest_path(src, dst, avoid_edges=avoid, cost_fn=cost_fn)



    def _aggr_evacuate_reverse(self,
                              edge: Tuple[int, int],
                              keeper: str,
                              sim_load: Dict[Tuple[int, int], float],
                              avoid_extra: Set[Tuple[int, int]],
                              moves_done_ref: List[int]) -> None:
        """Best-effort: move non-keeper flows off 'edge' (usually the reverse of the hot link).

        Runs after deciding to dedicate the hot link to the keeper and we want the reverse
        direction as clean as possible (not just locked).

        - Uses strict paths (does not invade other groups' locks)
        - Respects max_moves_per_run (global) if configured
        """
        cookies = list(self.flow_mgr.link_cookies.get(edge, set()))
        if not cookies:
            return

        def cookie_group(c: int) -> str:
            qid = self.flow_mgr.cookie_queue_id.get(c, None)
            return self.group_for_queue(qid)

        groups_on_edge = {cookie_group(c) for c in cookies}
        if len(groups_on_edge) <= 1:
            return

        keeper_g = str(keeper).strip().upper()
        to_move = [c for c in cookies if cookie_group(c) != keeper_g]
        if not to_move:
            return

        moved = 0
        failed = 0
        moved_by_group: Dict[str, int] = {}
        failed_by_group: Dict[str, int] = {}

        for c in to_move:
            # Budget global
            if self.max_moves_per_run and (moves_done_ref[0] >= self.max_moves_per_run):
                break

            g = cookie_group(c)
            desc = self.flow_mgr.cookie_desc.get(c)
            if desc is None:
                failed += 1
                failed_by_group[g] = failed_by_group.get(g, 0) + 1
                continue

            old_path = self.flow_mgr.cookie_path.get(c, [])
            p = self._strict_path(desc.src_dpid, desc.dst_dpid, g, sim_load, avoid_extra=set(avoid_extra), require_home=False)
            if not p or p == old_path:
                failed += 1
                failed_by_group[g] = failed_by_group.get(g, 0) + 1
                continue

            new_cookie = self.flow_mgr.reroute_cookie(c, p, table_id=self.table_id)
            if new_cookie is None:
                failed += 1
                failed_by_group[g] = failed_by_group.get(g, 0) + 1
                continue

            moved += 1
            moved_by_group[g] = moved_by_group.get(g, 0) + 1
            moves_done_ref[0] += 1

        if self.log_enabled and (moved or failed):
            self.log.info(
                "[TE][AGGR] reverse_evacuate edge=%s keeper=%s moved=%d failed=%d budget_left=%s",
                edge,
                keeper_g,
                moved,
                failed,
                (self.max_moves_per_run - moves_done_ref[0]) if self.max_moves_per_run else "inf",
            )
            if moved_by_group:
                self.log.info("[TE][AGGR] reverse_evacuate moved_by_group=%s", moved_by_group)
            if failed_by_group:
                self.log.info("[TE][AGGR] reverse_evacuate failed_by_group=%s", failed_by_group)

    # Original TE_QOS_C mode (unchanged logic)
    # ---------------------------------------------------------------------
    def _mlu(self, sim_load: Dict[Tuple[int, int], float]) -> float:
        m = 0.0
        for e, L in sim_load.items():
            C = self.stats.capacity_mbps(e)
            if C > 0:
                m = max(m, float(L) / C)
        return m

    def run_once(self):
        if self.te_mode == MODE_TE_QOS_D:
            return self._run_once_te_qos_d()
        return self._run_once_te_qos_c()

    def _run_once_te_qos_c(self):
        # Snapshot current loads
        sim_load: Dict[Tuple[int, int], float] = self.stats.snapshot_loads()
        mlu_before = self._mlu(sim_load)
        if self.log_enabled:
            self.log.info("[TE] start mlu_before=%.3f links=%d", mlu_before, len(sim_load))

        # Build hot links list
        hot_links = []
        for e, _Lhat in sim_load.items():
            if self.stats.is_hot(e):
                U = self.stats.util(e)
                hot_links.append((e, U))
        hot_links.sort(key=lambda x: x[1], reverse=True)
        if self.log_enabled:
            if hot_links:
                self.log.info("[TE] hot_links=%s", hot_links)
            else:
                self.log.info("[TE] no hot links; exit")

        for (hot_e, Uhot) in hot_links:
            if Uhot <= self.hot_th:
                continue

            all_cookies = list(self.flow_mgr.link_cookies.get(hot_e, set()))
            self.log.info("[TE] hot=%s U=%.3f cookies=%d", hot_e, Uhot, len(all_cookies))
            if not all_cookies:
                if self.log_enabled:
                    self.log.info("[TE] hot=%s U=%.3f no cookies on link", hot_e, Uhot)
                continue

            candidates = []
            for c in all_cookies:
                cls = str(self.flow_mgr.cookie_class.get(c, "")).strip().upper()
                if self.managed_classes is not None and cls not in self.managed_classes:
                    continue

                qid = self.flow_mgr.cookie_queue_id.get(c, None)

                if qid is None:
                    if self.unknown_queue_behavior == "lowest":
                        qid_eff = 10 ** 9
                    else:
                        continue
                else:
                    qid_eff = int(qid)

                if qid_eff in self.protect_queues:
                    continue

                rate = float(self.flow_mgr.cookie_rate_mbps.get(c, 0.0))
                candidates.append((qid_eff, rate, c))

            if not candidates:
                if self.log_enabled:
                    self.log.info(
                        "[TE] hot=%s U=%.3f no movable cookies (protect_queues=%s managed_classes=%s)",
                        hot_e,
                        Uhot,
                        sorted(self.protect_queues),
                        sorted(self.managed_classes) if self.managed_classes is not None else "ALL",
                    )
                continue

            # sort: low priority first
            if self.lower_queue_is_higher_priority:
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)  # qid desc, rate desc
            else:
                candidates.sort(key=lambda x: (x[0], x[1]))  # qid asc, rate asc

            if self.log_enabled:
                topk = [(hex(c), q, round(r, 3)) for (q, r, c) in candidates[:10]]
                self.log.info("[TE] hot=%s candidates(top10)=%s", hot_e, topk)

            for (qid_eff, _rate0, cookie) in candidates:
                last = self.flow_mgr.cookie_last_move.get(cookie, 0.0)
                if now() - last < self.cooldown_s:
                    if self.log_enabled:
                        self.log.info(
                            "[TE] cookie=%s skip cooldown remaining=%.1fs",
                            hex(cookie),
                            max(0.0, self.cooldown_s - (now() - last)),
                        )
                    continue

                desc = self.flow_mgr.cookie_desc.get(cookie)
                if desc is None:
                    if self.log_enabled:
                        self.log.info("[TE] cookie=%s no desc, skip", hex(cookie))
                    continue

                r = max(self.flow_mgr.cookie_rate_mbps.get(cookie, 0.0), self.r_min_mbps)
                r_need = self.safety_factor * r

                avoid = {hot_e, (hot_e[1], hot_e[0])}

                def cost_fn(e):  # Dijkstra cost function with utilization
                    C = self.stats.capacity_mbps(e)
                    L = sim_load.get(e, self.stats.link_load_mbps.get(e, 0.0))
                    U = (L / C) if C > 0 else 0.0
                    D = self.default_link_capacity / C if C > 0 else float("inf")
                    return self.Kc * D + self.Ku * U

                new_path = self.path_engine.shortest_path(
                    desc.src_dpid,
                    desc.dst_dpid,
                    avoid_edges=avoid,
                    min_residual_mbps=r_need,
                    cost_fn=cost_fn,
                )
                if not new_path:
                    if self.log_enabled:
                        self.log.info("[TE] cookie=%s no alt path (r_need=%.3f)", hex(cookie), r_need)
                    continue

                old_edges = self.flow_mgr.cookie_edges.get(cookie, set())
                new_edges = set((new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1))

                # what-if simulate
                sim2 = dict(sim_load)
                for e in old_edges:
                    sim2[e] = max(0.0, sim2.get(e, 0.0) - r)
                for e in new_edges:
                    sim2[e] = sim2.get(e, 0.0) + r

                cap_hot = self.stats.capacity_mbps(hot_e)
                Uhot_before = (sim_load.get(hot_e, 0.0) / cap_hot) if cap_hot > 0 else 0.0
                Uhot_after = (sim2.get(hot_e, 0.0) / cap_hot) if cap_hot > 0 else 0.0
                if (Uhot_before - Uhot_after) < self.delta_hot:
                    if self.log_enabled:
                        self.log.info(
                            "[TE] cookie=%s alt path insufficient cooling %.3f->%.3f",
                            hex(cookie),
                            Uhot_before,
                            Uhot_after,
                        )
                    continue

                mlu_after = self._mlu(sim2)
                if mlu_after > mlu_before + self.delta_global:
                    if self.log_enabled:
                        self.log.info(
                            "[TE] cookie=%s mlu would worsen %.3f->%.3f",
                            hex(cookie),
                            mlu_before,
                            mlu_after,
                        )
                    continue

                # APPLY
                old_mlu = mlu_before
                new_cookie = self.flow_mgr.reroute_cookie(cookie, new_path, table_id=self.table_id)
                if new_cookie is None:
                    if self.log_enabled:
                        self.log.warning("[TE] reroute failed cookie=%s path=%s", hex(cookie), new_path)
                    continue
                self.flow_mgr.cookie_last_move[new_cookie] = now()

                if self.log_enabled:
                    self.log.info(
                        "[TE] decision cookie=%s->%s hot_link=%s Uhot=%.3f->%.3f mlu=%.3f->%.3f",
                        hex(cookie),
                        hex(new_cookie),
                        hot_e,
                        Uhot_before,
                        Uhot_after,
                        old_mlu,
                        mlu_after,
                    )

                sim_load = sim2
                mlu_before = mlu_after

                # stop moving more flows for this hot link if it cooled enough
                Uhot_new = (sim_load.get(hot_e, 0.0) / cap_hot) if cap_hot > 0 else 0.0
                if Uhot_new <= (self.hot_th - 0.02):
                    break

    # ---------------------------------------------------------------------
    # TE_QOS_D mode
    # ---------------------------------------------------------------------
    def _run_once_te_qos_d(self):
        self._cleanup_locks()

        sim_load: Dict[Tuple[int, int], float] = self.stats.snapshot_loads()

        # Build hot links list
        hot_links: List[Tuple[Tuple[int, int], float]] = []
        for e in sim_load.keys():
            if self.stats.is_hot(e):
                U = self.stats.util(e)
                if U > self.hot_th:
                    hot_links.append((e, U))
        hot_links.sort(key=lambda x: x[1], reverse=True)

        if self.log_enabled:
            if hot_links:
                self.log.info("[TE][AGGR] hot_links=%s", hot_links)
            else:
                self.log.info("[TE][AGGR] no hot links; exit")

        moves_done = 0

        for (hot_e, Uhot) in hot_links:
            cookies = list(self.flow_mgr.link_cookies.get(hot_e, set()))
            if not cookies:
                continue

            rev = (hot_e[1], hot_e[0])

            def cookie_group(c: int) -> str:
                qid = self.flow_mgr.cookie_queue_id.get(c, None)
                return self.group_for_queue(qid)

            def edge_groups(edge: Tuple[int, int]) -> Set[str]:
                cs = self.flow_mgr.link_cookies.get(edge, set())
                if not cs:
                    return set()
                return {cookie_group(c) for c in cs}

            def edge_queue_ids(edge: Tuple[int, int]) -> List[Optional[int]]:
                cs = self.flow_mgr.link_cookies.get(edge, set())
                return sorted({self.flow_mgr.cookie_queue_id.get(c, None) for c in cs})

            # Rule: if the hot link is already dedicated to a single group, do not reroute.
            groups_on_hot = edge_groups(hot_e)
            if len(groups_on_hot) <= 1:
                g = next(iter(groups_on_hot)) if groups_on_hot else None
                if self.log_enabled:
                    self.log.info(
                        "[TE][AGGR] hot=%s U=%.3f single_group=%s queues=%s -> skip",
                        hot_e, Uhot, g, edge_queue_ids(hot_e)
                    )
                # If there is no lock, set it so packet_in prefers it as "home".
                if g and self._active_lock_group(hot_e) is None:
                    self.set_lock(hot_e, g)
                continue

            # --------------------------
            # Phase 1: If a group on the link already has a home lock, move it to its home (strict)
            # --------------------------
            home_groups = [g for g in groups_on_hot if self.group_has_home(g)]
            home_groups.sort(key=lambda g: self.group_rank(g))  # higher priority first

            if self.log_enabled:
                self.log.info(
                    "[TE][AGGR] hot=%s U=%.3f queues=%s groups=%s home_groups=%s locks=%s",
                    hot_e,
                    Uhot,
                    edge_queue_ids(hot_e),
                    sorted(list(groups_on_hot), key=lambda g: self.group_rank(g)),
                    home_groups,
                    {e: self._active_lock_group(e) for e in sorted(list(self.active_locked_edges()))},
                )

            for g in home_groups:
                # If the hot link is already locked for this group, skip it in phase1.
                if self._active_lock_group(hot_e) == str(g).strip().upper():
                    continue

                # Recompute cookies on the hot link (reroutes may have changed them).
                cur = list(self.flow_mgr.link_cookies.get(hot_e, set()))
                cur_g = [c for c in cur if cookie_group(c) == g]
                if not cur_g:
                    continue

                moved_g = 0
                for c in cur_g:
                    desc = self.flow_mgr.cookie_desc.get(c)
                    if desc is None:
                        continue
                    old_path = self.flow_mgr.cookie_path.get(c, [])
                    new_path = self._strict_path(
                        desc.src_dpid, desc.dst_dpid, g, sim_load,
                        avoid_extra={hot_e, rev}, require_home=True
                    )
                    if not new_path or new_path == old_path:
                        continue
                    new_cookie = self.flow_mgr.reroute_cookie(c, new_path, table_id=self.table_id)
                    if new_cookie is None:
                        continue
                    moved_g += 1
                    moves_done += 1

                if self.log_enabled and moved_g:
                    self.log.info("[TE][AGGR] hot=%s phase1 moved group=%s moves=%d", hot_e, g, moved_g)

            # After phase1, re-evaluate dedication
            groups_after = edge_groups(hot_e)
            if len(groups_after) <= 1 and groups_after:
                remaining_group = next(iter(groups_after))
                if self.log_enabled:
                    self.log.info(
                        "[TE][AGGR] hot=%s after phase1 now_single_group=%s queues=%s",
                        hot_e, remaining_group, edge_queue_ids(hot_e)
                    )
                self.set_lock(hot_e, remaining_group)
                continue

            # --------------------------
            # Phase 2: Dedicate the hot link to a "keeper" group
            # --------------------------
            cookies2 = list(self.flow_mgr.link_cookies.get(hot_e, set()))
            if not cookies2:
                continue

            groups2 = {cookie_group(c) for c in cookies2}
            if len(groups2) <= 1:
                remaining_group = next(iter(groups2))
                if self.log_enabled:
                    self.log.info(
                        "[TE][AGGR] hot=%s after phase1 already_single_group=%s queues=%s",
                        hot_e, remaining_group, edge_queue_ids(hot_e)
                    )
                self.set_lock(hot_e, remaining_group)
                continue

            # If the link already has an active lock for a present group, that group is always the keeper.
            locked_here = self._active_lock_group(hot_e)
            if locked_here is not None and locked_here in groups2:
                keeper = locked_here
            else:
                keeper = min(groups2, key=lambda g: self.group_rank(g))  # highest priority

            to_move = [c for c in cookies2 if cookie_group(c) != keeper]
            if not to_move:
                continue

            # Feasibility: we must be able to move all non-keeper flows (without invading foreign locks).
            candidate_paths: Dict[int, List[int]] = {}
            feasible = True
            for c in to_move:
                g = cookie_group(c)
                desc = self.flow_mgr.cookie_desc.get(c)
                if desc is None:
                    feasible = False
                    break
                p = self._strict_path(
                    desc.src_dpid, desc.dst_dpid, g, sim_load,
                    avoid_extra={hot_e, rev}, require_home=False
                )
                if not p:
                    feasible = False
                    break
                candidate_paths[c] = p

            if not feasible:
                if self.log_enabled:
                    self.log.info(
                        "[TE][AGGR] hot=%s phase2 abort: cannot move all non-keeper groups (keeper=%s)",
                        hot_e, keeper
                    )
                continue

            # Move order: lowest priority first, higher rate first
            def move_sort_key(c: int):
                g = cookie_group(c)
                r = float(self.flow_mgr.cookie_rate_mbps.get(c, 0.0))
                return (self.group_rank(g), -r)

            to_move_sorted = sorted(to_move, key=move_sort_key, reverse=True)

            # Budget (if any). Note: phase1 ignores budget.
            if self.max_moves_per_run and (moves_done >= self.max_moves_per_run):
                if self.log_enabled:
                    self.log.info("[TE][AGGR] budget exhausted max_moves_per_run=%d", self.max_moves_per_run)
                break

            budget_left = (self.max_moves_per_run - moves_done) if self.max_moves_per_run else len(to_move_sorted)
            moving_now = to_move_sorted[:budget_left] if self.max_moves_per_run else to_move_sorted

            # ---- Temporary "pending lock" to prevent packet_in from reinstalling non-keeper flows
            # over the hot link while migration is in progress.
            pending_created: Optional[float] = None
            pending_ttl = min(2.0, max(0.5, float(self.te_period)))
            if self.link_lock.get(hot_e) is None:
                t0 = now()
                pending_created = t0
                self.link_lock[hot_e] = _LockEntry(group=str(keeper).strip().upper(), created_at=t0, expires_at=t0 + pending_ttl)
                if self.lock_bidirectional:
                    self.link_lock[rev] = _LockEntry(group=str(keeper).strip().upper(), created_at=t0, expires_at=t0 + pending_ttl)

            committed = False
            moved2 = 0
            failed2 = 0
            moved_by_group: Dict[str, int] = {}
            failed_by_group: Dict[str, int] = {}

            try:
                for c in moving_now:
                    g = cookie_group(c)
                    p = candidate_paths.get(c)
                    if not p:
                        failed2 += 1
                        failed_by_group[g] = failed_by_group.get(g, 0) + 1
                        continue
                    old_path = self.flow_mgr.cookie_path.get(c, [])
                    if p == old_path:
                        # Should not happen (avoid hot_e); treat as failure to avoid locking.
                        failed2 += 1
                        failed_by_group[g] = failed_by_group.get(g, 0) + 1
                        continue
                    new_cookie = self.flow_mgr.reroute_cookie(c, p, table_id=self.table_id)
                    if new_cookie is None:
                        failed2 += 1
                        failed_by_group[g] = failed_by_group.get(g, 0) + 1
                        continue
                    moved2 += 1
                    moved_by_group[g] = moved_by_group.get(g, 0) + 1
                    moves_done += 1

                if self.log_enabled:
                    self.log.info(
                        "[TE][AGGR] hot=%s phase2 attempted=%d moved=%d failed=%d keeper=%s remaining_budget=%s",
                        hot_e,
                        len(moving_now),
                        moved2,
                        failed2,
                        keeper,
                        (self.max_moves_per_run - moves_done) if self.max_moves_per_run else "inf",
                    )
                    if moved_by_group:
                        self.log.info("[TE][AGGR] hot=%s phase2 moved_by_group=%s", hot_e, moved_by_group)
                    if failed_by_group:
                        self.log.info("[TE][AGGR] hot=%s phase2 failed_by_group=%s", hot_e, failed_by_group)

                # Attempt to lock only if:
                #  - No budget cut (moving_now == to_move_sorted)
                #  - No failures (moved2 == len(moving_now))
                attempted_all = (len(moving_now) == len(to_move_sorted))
                success_all = attempted_all and (moved2 == len(moving_now)) and (failed2 == 0)

                if success_all:
                    groups3 = edge_groups(hot_e)
                    if groups3 == {keeper}:
                        self.set_lock(hot_e, keeper)
                        committed = True
                        # If lock_bidirectional is active, also clean the reverse direction
                        # (move non-keeper flows off the reverse), not just lock it.
                        if self.lock_bidirectional:
                            md_ref = [moves_done]
                            self._aggr_evacuate_reverse(rev, keeper, sim_load, avoid_extra={hot_e, rev}, moves_done_ref=md_ref)
                            moves_done = md_ref[0]

                    else:
                        if self.log_enabled:
                            self.log.info(
                                "[TE][AGGR] hot=%s phase2 success_all but still multi_group=%s queues=%s -> no lock",
                                hot_e,
                                sorted(list(groups3), key=lambda g: self.group_rank(g)),
                                edge_queue_ids(hot_e),
                            )
                else:
                    if self.log_enabled:
                        if not attempted_all:
                            self.log.info(
                                "[TE][AGGR] hot=%s phase2 partial_due_budget=%d/%d -> lock deferred",
                                hot_e, len(moving_now), len(to_move_sorted)
                            )
                        else:
                            self.log.info(
                                "[TE][AGGR] hot=%s phase2 not_all_moved=%d/%d -> lock deferred",
                                hot_e, moved2, len(to_move_sorted)
                            )

            finally:
                # Remove pending lock if it was not consolidated with set_lock()
                if pending_created is not None and not committed:
                    ent = self.link_lock.get(hot_e)
                    if ent and ent.created_at == pending_created and ent.group == str(keeper).strip().upper():
                        self.link_lock.pop(hot_e, None)
                    if self.lock_bidirectional:
                        ent2 = self.link_lock.get(rev)
                        if ent2 and ent2.created_at == pending_created and ent2.group == str(keeper).strip().upper():
                            self.link_lock.pop(rev, None)

        return

    # Legacy aliases (backward compatibility with old names)
    _run_once_conditional = _run_once_te_qos_c
    _run_once_aggressive = _run_once_te_qos_d
