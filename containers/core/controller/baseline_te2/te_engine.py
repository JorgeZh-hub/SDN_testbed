# te_engine.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Set

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
                 r_min_mbps: float = 0.1):
        self.topo = topo
        self.stats = stats
        self.path_engine = path_engine
        self.flow_mgr = flow_mgr
        self.log = logger

        self.te_period = float(te_period)
        self.cooldown_s = float(cooldown_s)
        self.hot_th = float(hot_th)
        self.delta_hot = float(delta_hot)
        self.delta_global = float(delta_global)
        self.K = float(K)
        self.safety_factor = float(safety_factor)
        self.r_min_mbps = float(r_min_mbps)

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

        # Build hot links list
        hot_links = []
        for e, Lhat in sim_load.items():
            if self.stats.is_hot(e):
                U = self.stats.util(e)
                hot_links.append((e, U))
        hot_links.sort(key=lambda x: x[1], reverse=True)

        for (hot_e, Uhot) in hot_links:
            if Uhot <= self.hot_th:
                continue

            crit_cookies = [c for c in self.flow_mgr.link_cookies.get(hot_e, set())
                            if self.flow_mgr.cookie_class.get(c) == "CRIT"]
            if not crit_cookies:
                continue

            crit_cookies.sort(key=lambda c: self.flow_mgr.cookie_rate_mbps.get(c, 0.0), reverse=True)

            for cookie in crit_cookies:
                last = self.flow_mgr.cookie_last_move.get(cookie, 0.0)
                if now() - last < self.cooldown_s:
                    continue

                desc = self.flow_mgr.cookie_desc.get(cookie)
                if desc is None:
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
                    continue

                mlu_after = self._mlu(sim2)
                if mlu_after > mlu_before + self.delta_global:
                    continue

                # APPLY
                old_mlu = mlu_before
                new_cookie = self.flow_mgr.reroute_cookie(cookie, new_path)
                if new_cookie is None:
                    continue
                self.flow_mgr.cookie_last_move[new_cookie] = now()

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
