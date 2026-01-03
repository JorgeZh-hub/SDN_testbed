# te_engine.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional

from .utils import now

class TEEngine:
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
                 K: float = 5.0,
                 safety_factor: float = 1.2,
                 r_min_mbps: float = 0.1,
                 table_id: int = 0,
                 managed_classes: Optional[List[str]] = None,
                 protect_queues: Optional[List[int]] = None,
                 lower_queue_is_higher_priority: bool = True,
                 unknown_queue_behavior: str = "protect",
                 log_enabled: bool = True):
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
        self.K = float(K)
        self.safety_factor = float(safety_factor)
        self.r_min_mbps = float(r_min_mbps)
        # Filtro opcional por clase: si managed_classes es None/vacío => considerar todas.
        if managed_classes is None:
            self.managed_classes = None
        else:
            mc = {str(x).strip().upper() for x in managed_classes if str(x).strip()}
            self.managed_classes = mc if mc else None

        # Protección por cola (no mover estos flows). Default: proteger la cola "más prioritaria" 0.
        if protect_queues is None:
            self.protect_queues = {0}
        else:
            self.protect_queues = {int(x) for x in protect_queues}

        # Convención de prioridad:
        # - Si lower_queue_is_higher_priority=True (default): queue_id menor => mayor prioridad.
        #   Entonces TE empieza moviendo queue_id mayor (menos prioridad).
        self.lower_queue_is_higher_priority = bool(lower_queue_is_higher_priority)

        # Si no se conoce la cola de un cookie (None):
        # - "protect" (default): no lo mueve
        # - "lowest": lo trata como el de menor prioridad (lo mueve primero)
        self.unknown_queue_behavior = str(unknown_queue_behavior or "protect").strip().lower()
        if self.unknown_queue_behavior not in ("protect", "lowest"):
            self.unknown_queue_behavior = "protect"

    def _mlu(self, sim_load: Dict[Tuple[int,int], float]) -> float:
        m = 0.0
        for e, L in sim_load.items():
            C = self.stats.capacity_mbps(e)
            if C > 0:
                m = max(m, float(L)/C)
        return m

    def run_once(self):
        # Snapshot current loads
        sim_load: Dict[Tuple[int,int], float] = self.stats.snapshot_loads()
        mlu_before = self._mlu(sim_load)
        if self.log_enabled:
            self.log.info("[TE] start mlu_before=%.3f links=%d", mlu_before, len(sim_load))

        # Build hot links list
        hot_links = []
        for e, Lhat in sim_load.items():
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

            # Candidatos en el enlace caliente:
            # - Orden: desde MENOS prioritarios (cola más alta) hacia más prioritarios
            # - Dentro de cada prioridad: mayor BW primero
            all_cookies = list(self.flow_mgr.link_cookies.get(hot_e, set()))
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
                        # tratar como el de menor prioridad
                        qid_eff = 10**9
                    else:
                        # protect
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
                        hot_e, Uhot, sorted(self.protect_queues),
                        sorted(self.managed_classes) if self.managed_classes is not None else "ALL"
                    )
                continue

            # sort: low priority first
            # lower_queue_is_higher_priority=True => larger qid is lower priority => sort qid desc
            if self.lower_queue_is_higher_priority:
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)  # qid desc, rate desc
            else:
                candidates.sort(key=lambda x: (x[0], x[1]))  # qid asc, rate asc

            if self.log_enabled:
                # log top few candidates
                topk = [(hex(c), q, round(r, 3)) for (q, r, c) in candidates[:10]]
                self.log.info("[TE] hot=%s candidates(top10)=%s", hot_e, topk)

            for (qid_eff, rate0, cookie) in candidates:
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

                def cost_fn(e):
                    # cost = 1 + K*U
                    C = self.stats.capacity_mbps(e)
                    L = sim_load.get(e, self.stats.link_load_mbps.get(e, 0.0))
                    U = (L / C) if C > 0 else 0.0
                    return 1.0 + self.K * U

                new_path = self.path_engine.shortest_path(
                    desc.src_dpid, desc.dst_dpid,
                    avoid_edges=avoid,
                    min_residual_mbps=r_need,
                    cost_fn=cost_fn
                )
                if not new_path:
                    if self.log_enabled:
                        self.log.info("[TE] cookie=%s no alt path (r_need=%.3f)", hex(cookie), r_need)
                    continue

                old_edges = self.flow_mgr.cookie_edges.get(cookie, set())
                new_edges = set((new_path[i], new_path[i+1]) for i in range(len(new_path)-1))

                # what-if simulate
                sim2 = dict(sim_load)
                for e in old_edges:
                    sim2[e] = max(0.0, sim2.get(e, 0.0) - r)
                for e in new_edges:
                    sim2[e] = sim2.get(e, 0.0) + r

                Uhot_before = (sim_load.get(hot_e, 0.0) / self.stats.capacity_mbps(hot_e)) if self.stats.capacity_mbps(hot_e)>0 else 0.0
                Uhot_after = (sim2.get(hot_e, 0.0) / self.stats.capacity_mbps(hot_e)) if self.stats.capacity_mbps(hot_e)>0 else 0.0
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
                Uhot_new = (sim_load.get(hot_e, 0.0) / self.stats.capacity_mbps(hot_e)) if self.stats.capacity_mbps(hot_e)>0 else 0.0
                if Uhot_new <= (self.hot_th - 0.02):
                    break
