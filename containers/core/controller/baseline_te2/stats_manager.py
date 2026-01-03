# stats_manager.py
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

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
                 log_enabled: bool = True,
                 csv_dir: Optional[str] = None):
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
        self._prev_rx_drop = defaultdict(lambda: defaultdict(int))
        self._prev_tx_drop = defaultdict(lambda: defaultdict(int))

        # directed-link metrics
        self.link_load_mbps = defaultdict(float)  # e=(u,v)->Lhat
        self.link_capacity_mbps = defaultdict(lambda: self.default_capacity_mbps)
        self.hot_counter = defaultdict(int)       # e->consecutive

        # flow counter history: dpid->cookie->(bytes,ts)
        self._prev_cookie_bytes = defaultdict(dict)
        self._prev_cookie_ts = defaultdict(dict)

        # thread handles (optional)
        self._running = False

        # CSV export (minimal, no extra config)
        if csv_dir:
            self._csv_dir = os.path.abspath(csv_dir)
        else:
            self._csv_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src", "experiments")
            )
        try:
            os.makedirs(self._csv_dir, exist_ok=True)
        except OSError:
            self._csv_dir = "."
        self._link_csv_path = os.path.join(self._csv_dir, "link_stats.csv")
        self._flow_csv_path = os.path.join(self._csv_dir, "flow_stats.csv")
        try:
            self._link_csv_header_written = os.path.exists(self._link_csv_path) and os.path.getsize(self._link_csv_path) > 0
        except OSError:
            self._link_csv_header_written = False
        try:
            self._flow_csv_header_written = os.path.exists(self._flow_csv_path) and os.path.getsize(self._flow_csv_path) > 0
        except OSError:
            self._flow_csv_header_written = False

        self._cookie_rate_by_dp = defaultdict(dict)
        self._cookie_match = {}

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
        csv_file = None
        csv_writer = None
        csv_file = open(self._link_csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if not self._link_csv_header_written:
            csv_writer.writerow([
                "timestamp",
                "src_dpid",
                "dst_dpid",
                "port_no",
                "mbps",
                "rx_drop",
                "tx_drop",
                "total_drop",
                "util_pct",
            ])
            self._link_csv_header_written = True
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
            rx_drop = int(getattr(stat, "rx_dropped", 0))
            tx_drop = int(getattr(stat, "tx_dropped", 0))
            prev_rx_drop = self._prev_rx_drop[dpid][port_no]
            prev_tx_drop = self._prev_tx_drop[dpid][port_no]
            rx_drop_delta = max(0, rx_drop - prev_rx_drop)
            tx_drop_delta = max(0, tx_drop - prev_tx_drop)
            total_drop = rx_drop_delta + tx_drop_delta
            rx_drop_rate = rx_drop_delta / dt
            tx_drop_rate = tx_drop_delta / dt
            total_drop_rate = total_drop / dt

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
            self._prev_rx_drop[dpid][port_no] = rx_drop
            self._prev_tx_drop[dpid][port_no] = tx_drop
            updated += 1
            if csv_writer is not None:
                cap = self.capacity_mbps(e)
                u_inst = (mbps / cap) if cap > 0 else 0.0
                csv_writer.writerow([
                    f"{ts:.6f}",
                    dpid,
                    v,
                    port_no,
                    f"{mbps:.6f}",
                    f"{rx_drop_rate:.3f}",
                    f"{tx_drop_rate:.3f}",
                    f"{total_drop_rate:.3f}",
                    f"{u_inst * 100.0:.3f}",
                ])
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
        if csv_file is not None:
            csv_file.close()
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
        dpid = dp.id
        # We'll estimate cookie rate by byte delta / dt and EWMA.
        # Important: stats come per-switch; we conservatively take max across switches in FlowManager.
        updated = 0
        csv_file = None
        csv_writer = None
        csv_file = open(self._flow_csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if not self._flow_csv_header_written:
            csv_writer.writerow([
                "timestamp",
                "cookie",
                "match",
                "avg_mbps",
            ])
            self._flow_csv_header_written = True
        for st in msg.body:
            cookie = int(st.cookie)
            if self.flow_mgr.cookie_path.get(cookie, [0])[-1] != dpid:
                continue  # only consider flows that end at this dp
            byte_count = int(getattr(st, "byte_count", 0))
            prev_b = self._prev_cookie_bytes[dpid].get(cookie, None)
            prev_t = self._prev_cookie_ts[dpid].get(cookie, None)
            if prev_b is None or prev_t is None:
                self._prev_cookie_bytes[dpid][cookie] = byte_count
                self._prev_cookie_ts[dpid][cookie] = ts
                if self.log_enabled:
                    self.log.info(
                        "[STATS] flow_init dpid=%s cookie=%s bytes=%s ts=%.6f",
                        dpid,
                        hex(cookie),
                        byte_count,
                        ts,
                    )
                continue
            dt = max(1e-3, ts - prev_t)
            if dt <= self.flowstats_period * 0.2:
                continue  # too soon since last update
            delta = max(0, byte_count - prev_b)
            mbps = (delta * 8.0 / 1e6) / dt

            self._prev_cookie_bytes[dpid][cookie] = byte_count
            self._prev_cookie_ts[dpid][cookie] = ts

            old_rate = self.flow_mgr.cookie_rate_mbps.get(cookie, 0.0)
            ewma = self.alpha_flow * mbps + (1.0 - self.alpha_flow) * old_rate

            # feed FlowManager
            self.flow_mgr.update_cookie_rate(cookie, mbps, alpha=self.alpha_flow)
            new_rate = self.flow_mgr.cookie_rate_mbps.get(cookie, 0.0)
            updated += 1
            avg_mbps = None
            rates_n = 0
            stale_n = 0
            if csv_writer is not None:
                self._cookie_rate_by_dp[cookie][dpid] = (mbps, ts)
                stale = [
                    k for k, (_, t_last) in self._cookie_rate_by_dp[cookie].items()
                    if ts - t_last > (self.flowstats_period * 2.0)
                ]
                stale_n = len(stale)
                for k in stale:
                    self._cookie_rate_by_dp[cookie].pop(k, None)
                rates = [v for v, _ in self._cookie_rate_by_dp[cookie].values()]
                rates_n = len(rates)
                avg_mbps = (sum(rates) / len(rates)) if rates else 0.0
                match_str = str(getattr(st, "match", "-"))
                self._cookie_match[cookie] = match_str
                csv_writer.writerow([
                    f"{ts:.6f}",
                    hex(cookie),
                    match_str,
                    f"{avg_mbps:.6f}",
                ])
            if self.log_enabled:
                cls = self.flow_mgr.cookie_class.get(cookie, "-")
                path = self.flow_mgr.cookie_path.get(cookie, [])
                avg_mbps_str = f"{avg_mbps:.3f}" if avg_mbps is not None else "-"
                self.log.info(
                    "[STATS] flow dpid=%s cookie=%s class=%s mbps=%.3f dt=%.3f deltaB=%s prevB=%s bytes=%s ewma=%.3f rate=%.3f avg_mbps=%s rates_n=%s stale=%s path=%s",
                    dpid,
                    hex(cookie),
                    cls,
                    mbps,
                    dt,
                    delta,
                    prev_b,
                    byte_count,
                    ewma,
                    new_rate,
                    avg_mbps_str,
                    rates_n,
                    stale_n,
                    path,
                )
        if csv_file is not None:
            csv_file.close()
        if self.log_enabled and updated:
            self.log.info("[STATS] flow_reply cookies=%s", updated)

    # -------- Helpers for TE --------
    def is_hot(self, e: Tuple[int,int]) -> bool:
        return self.hot_counter.get(e, 0) >= self.hot_n

    def snapshot_loads(self) -> Dict[Tuple[int,int], float]:
        return dict(self.link_load_mbps)
