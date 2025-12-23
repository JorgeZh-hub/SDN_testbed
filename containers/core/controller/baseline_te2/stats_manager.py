# stats_manager.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

from .utils import now

class StatsManager:
    """Observación de enlaces (PortStats) y flujos (FlowStats) para TE."""

    def __init__(self,
                 topo,
                 flow_mgr,
                 logger,
                 default_capacity_mbps: float = 100.0,
                 alpha_link: float = 0.3,
                 alpha_flow: float = 0.3,
                 hot_th: float = 0.85,
                 hot_n: int = 2,
                 portstats_period: float = 1.0,
                 flowstats_period: float = 5.0,
                 app_id: int = 0xBEEF,
                 log_enabled: bool = True):
        self.topo = topo
        self.flow_mgr = flow_mgr
        self.log = logger
        self.log_enabled = bool(log_enabled)

        self.default_capacity_mbps = float(default_capacity_mbps)
        self.alpha_link = float(alpha_link)
        self.alpha_flow = float(alpha_flow)
        self.hot_th = float(hot_th)
        self.hot_n = int(hot_n)

        self.portstats_period = float(portstats_period)
        self.flowstats_period = float(flowstats_period)
        self.app_id = int(app_id) & 0xFFFF

        # port counter history: dpid->port->(bytes,ts)
        self._prev_tx_bytes = defaultdict(lambda: defaultdict(int))
        self._prev_ts = defaultdict(lambda: defaultdict(float))

        # directed-link metrics
        self.link_load_mbps = defaultdict(float)  # e=(u,v)->Lhat
        self.link_capacity_mbps = defaultdict(lambda: self.default_capacity_mbps)
        self.hot_counter = defaultdict(int)       # e->consecutive

        # flow counter history: cookie->(bytes,ts)
        self._prev_cookie_bytes = defaultdict(int)
        self._prev_cookie_ts = defaultdict(float)

        # thread handles (optional)
        self._running = False

    # -------- capacities --------
    def set_capacity(self, e: Tuple[int,int], cap_mbps: float):
        self.link_capacity_mbps[e] = float(cap_mbps)

    def capacity_mbps(self, e: Tuple[int,int]) -> float:
        return float(self.link_capacity_mbps.get(e, self.default_capacity_mbps))

    def residual_mbps(self, e: Tuple[int,int]) -> float:
        return max(0.0, self.capacity_mbps(e) - float(self.link_load_mbps.get(e, 0.0)))

    def util(self, e: Tuple[int,int]) -> float:
        c = self.capacity_mbps(e)
        if c <= 0:
            return 0.0
        return float(self.link_load_mbps.get(e, 0.0)) / c

    # -------- PortStats request/reply integration --------
    def request_port_stats(self):
        for dp in list(self.topo.datapaths.values()):
            ofp = dp.ofproto
            parser = dp.ofproto_parser
            req = parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY)
            dp.send_msg(req)

    def handle_port_stats_reply(self, dp, msg):
        ts = now()
        dpid = dp.id
        updated = 0
        for stat in msg.body:
            port_no = stat.port_no
            # only consider ports that map to a neighbor (directed link)
            v = self.topo.port_to_neigh.get(dpid, {}).get(port_no)
            if v is None:
                continue

            prev_b = self._prev_tx_bytes[dpid][port_no]
            prev_t = self._prev_ts[dpid][port_no] or ts
            dt = max(1e-3, ts - prev_t)
            delta = max(0, int(stat.tx_bytes) - int(prev_b))
            mbps = (delta * 8.0 / 1e6) / dt

            e = (dpid, v)
            old = self.link_load_mbps.get(e, 0.0)
            new = self.alpha_link * mbps + (1.0 - self.alpha_link) * old
            self.link_load_mbps[e] = new

            # hot confirm counter
            u = self.util(e)
            if u > self.hot_th:
                self.hot_counter[e] += 1
            else:
                self.hot_counter[e] = 0

            self._prev_tx_bytes[dpid][port_no] = int(stat.tx_bytes)
            self._prev_ts[dpid][port_no] = ts
            updated += 1
            if self.log_enabled:
                port_name = self.topo.get_port_name(dpid, port_no) or ""
                self.log.info(
                    "[STATS] port dpid=%s port=%s(%s) neigh=%s tx_mbps=%.3f ewma=%.3f util=%.3f hot_cnt=%s",
                    dpid,
                    port_no,
                    port_name,
                    v,
                    mbps,
                    new,
                    u,
                    self.hot_counter.get(e, 0),
                )
        if self.log_enabled and updated:
            hot_now = {e: self.hot_counter[e] for e in self.hot_counter if self.hot_counter[e] > 0}
            self.log.info("[STATS] port_reply dpid=%s updated_ports=%s hot=%s", dpid, updated, hot_now)

    # -------- FlowStats request/reply integration --------
    def request_flow_stats(self):
        # Only our app cookies: mask top 16 bits (APP_ID)
        cookie = (self.app_id & 0xFFFF) << 48
        cookie_mask = (0xFFFF) << 48

        for dp in list(self.topo.datapaths.values()):
            parser = dp.ofproto_parser
            req = parser.OFPFlowStatsRequest(
                dp,
                0,
                dp.ofproto.OFPTT_ALL,
                dp.ofproto.OFPP_ANY,
                dp.ofproto.OFPG_ANY,
                cookie=cookie,
                cookie_mask=cookie_mask
            )
            dp.send_msg(req)

    def handle_flow_stats_reply(self, dp, msg):
        ts = now()
        # We'll estimate cookie rate by byte delta / dt and EWMA.
        # Important: stats come per-switch; we conservatively take max across switches in FlowManager.
        updated = 0
        for st in msg.body:
            cookie = int(st.cookie)
            byte_count = int(getattr(st, "byte_count", 0))
            prev_b = self._prev_cookie_bytes.get(cookie, None)
            prev_t = self._prev_cookie_ts.get(cookie, None)
            if prev_b is None or prev_t is None:
                self._prev_cookie_bytes[cookie] = byte_count
                self._prev_cookie_ts[cookie] = ts
                continue
            dt = max(1e-3, ts - prev_t)
            delta = max(0, byte_count - prev_b)
            mbps = (delta * 8.0 / 1e6) / dt

            self._prev_cookie_bytes[cookie] = byte_count
            self._prev_cookie_ts[cookie] = ts

            # feed FlowManager
            self.flow_mgr.update_cookie_rate(cookie, mbps, alpha=self.alpha_flow)
            updated += 1
            if self.log_enabled:
                cls = self.flow_mgr.cookie_class.get(cookie, "-")
                path = self.flow_mgr.cookie_path.get(cookie, [])
                self.log.info(
                    "[STATS] flow cookie=%s class=%s mbps=%.3f path=%s",
                    hex(cookie),
                    cls,
                    mbps,
                    path,
                )
        if self.log_enabled and updated:
            self.log.info("[STATS] flow_reply cookies=%s", updated)

    # -------- Helpers for TE --------
    def is_hot(self, e: Tuple[int,int]) -> bool:
        return self.hot_counter.get(e, 0) >= self.hot_n

    def snapshot_loads(self) -> Dict[Tuple[int,int], float]:
        return dict(self.link_load_mbps)
