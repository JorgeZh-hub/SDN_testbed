# flow_manager.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, List, Any


from .utils import FlowKey, FlowDescriptor, stable_u32_hash
from .classifier import PriorityClassifier

class FlowManager:
    def __init__(self,
                 topo,
                 logger,
                 app_id: int = 0xBEEF,
                 priority_base: int = 20000,
                 barrier: bool = True,
                 queues_enable: bool = False,
                 crit_queue_id: int = 1,
                 be_queue_id: int = 0,
                 classifier: Optional[PriorityClassifier] = None,
                 log_enabled: bool = True):
        self.topo = topo
        self.log = logger
        self.log_enabled = bool(log_enabled)
        self.app_id = int(app_id) & 0xFFFF
        self.priority_base = int(priority_base)
        self.use_barrier = bool(barrier)
        self.use_queues = bool(queues_enable)
        self.queue_id_crit = int(crit_queue_id)
        self.queue_id_be = int(be_queue_id)
        self.classifier = classifier or PriorityClassifier([])

        # ---- TE indexes ----
        self.cookie_class: Dict[int, str] = {}               # cookie -> CRIT/BE
        self.cookie_edges: Dict[int, Set[Tuple[int,int]]] = {}
        self.link_cookies: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
        self.cookie_last_move: Dict[int, float] = defaultdict(float)
        self.cookie_rate_mbps: Dict[int, float] = defaultdict(float)
        self.cookie_desc: Dict[int, FlowDescriptor] = {}
        self.cookie_path: Dict[int, List[int]] = {}          # cookie -> [dpids]

        # flow_key -> active cookie
        self._active_cookie_by_key: Dict[Any, int] = {}

        # Host location (first seen): mac -> (dpid, port)
        self.hosts: Dict[str, Tuple[int, int]] = {}


    # ---------- Cookie ----------
    def cookie_for_key(self, key: Any, version: int) -> int:
        h = stable_u32_hash(key)
        cookie = (self.app_id << 48) | ((h & 0xFFFFFFFF) << 16) | (version & 0xFFFF)
        return cookie & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def cookie_base(cookie: int) -> int:
        return cookie & 0xFFFFFFFFFFFF0000

    @staticmethod
    def cookie_version(cookie: int) -> int:
        return cookie & 0xFFFF

    def next_cookie(self, cookie: int) -> int:
        base = self.cookie_base(cookie)
        ver = (self.cookie_version(cookie) + 1) & 0xFFFF
        return base | ver

    def get_or_create_cookie(self, desc: FlowDescriptor) -> int:
        cls, stable_side, stable_port, rule = self.classify_detail(desc, desc.dscp)
        ckey = self._cookie_key(desc, cls, stable_side, stable_port)

        cur = self._active_cookie_by_key.get(ckey)
        if cur is None:
            cur = self.cookie_for_key(ckey, 0)
            self._active_cookie_by_key[ckey] = cur

            # Guardamos clase temprano (evita recalcular y ayuda a TE/stats)
            self.cookie_class[cur] = cls

            k = desc.key
            if self.log_enabled:
                self.log.info(
                    "[CLASS] new_cookie=%s class=%s rule=%s stable=%s:%s 5t=%s:%s -> %s:%s proto=%s",
                    hex(cur), cls, rule or "-",
                    stable_side or "-", stable_port if stable_port is not None else "-",
                    k.ip_src or "-", k.l4_src if k.l4_src is not None else "-",
                    k.ip_dst or "-", k.l4_dst if k.l4_dst is not None else "-",
                    k.ip_proto if k.ip_proto is not None else "-"
                )

        return cur


    # ---------- Classification ----------
    def classify(self, desc: FlowDescriptor, dscp: Optional[int]) -> str:
        k = desc.key
        return self.classifier.classify(
            src_mac=k.src_mac,
            dst_mac=k.dst_mac,
            ip_src=k.ip_src,
            ip_dst=k.ip_dst,
            ip_proto=k.ip_proto,
            l4_src=k.l4_src,
            l4_dst=k.l4_dst,
            dscp=dscp
        )

    def classify_detail(self, desc: FlowDescriptor, dscp: Optional[int]):
        """
        Retorna (cls, stable_side, stable_port, rule_name)
        stable_side: "src" | "dst" | None
        """
        k = desc.key
        if hasattr(self.classifier, "classify_detail"):
            return self.classifier.classify_detail(
                src_mac=k.src_mac,
                dst_mac=k.dst_mac,
                ip_src=k.ip_src,
                ip_dst=k.ip_dst,
                ip_proto=k.ip_proto,
                l4_src=k.l4_src,
                l4_dst=k.l4_dst,
                dscp=dscp
            )
        # fallback si por alguna razón no existe
        return self.classify(desc, dscp), None, None, None

    def _cookie_key(self, desc: FlowDescriptor, cls: str,
                    stable_side: Optional[str], stable_port: Optional[int]) -> Any:
        """
        Debe alinearse con el MATCH instalado.
        - BE: no incluye L4
        - CRIT con puerto estable: incluye (stable_side, stable_port)
        """
        k = desc.key
        base = (k.eth_type, k.src_mac, k.dst_mac, k.ip_src, k.ip_dst, k.ip_proto)

        if cls == "CRIT" and stable_side and stable_port and k.ip_proto in (6, 17):
            return base + (stable_side, int(stable_port))

        return base

    # ---------- Rate updates ----------
    def update_cookie_rate(self, cookie: int, observed_mbps: float, alpha: float = 0.3):
        # Conservative aggregation across switches: keep max EWMA we have seen
        old = self.cookie_rate_mbps.get(cookie, 0.0)
        ewma = alpha * float(observed_mbps) + (1.0 - alpha) * float(old)
        # across switches, call multiple times; keep max
        self.cookie_rate_mbps[cookie] = max(old, ewma)

    # ---------- Index maintenance ----------
    def _register_cookie_path(self, cookie: int, path: List[int]):
        edges = set()
        for i in range(len(path)-1):
            edges.add((path[i], path[i+1]))
        self.cookie_edges[cookie] = edges
        self.cookie_path[cookie] = list(path)
        for e in edges:
            self.link_cookies[e].add(cookie)

    def _unregister_cookie(self, cookie: int):
        edges = self.cookie_edges.pop(cookie, set())
        for e in edges:
            s = self.link_cookies.get(e)
            if s:
                s.discard(cookie)
                if not s:
                    self.link_cookies.pop(e, None)
        self.cookie_path.pop(cookie, None)
        self.cookie_desc.pop(cookie, None)
        self.cookie_class.pop(cookie, None)
        self.cookie_rate_mbps.pop(cookie, None)
        self.cookie_last_move.pop(cookie, None)

    # ---------- OpenFlow operations ----------
    def _send_barrier(self, dp):
        if not self.use_barrier:
            return
        parser = dp.ofproto_parser
        dp.send_msg(parser.OFPBarrierRequest(dp))

    def install_path(self,
                     desc: FlowDescriptor,
                     cookie: int,
                     path: List[int],
                     last_port: int,
                     idle_timeout: int = 0,
                     hard_timeout: int = 0,
                     table_id: int = 0):
        """Instala reglas en cada switch del path (direccion src->dst)."""
        if not path:
            return

        # Build match from FlowKey
        match_kwargs = {}
        k = desc.key
        if k.eth_type is not None:
            match_kwargs["eth_type"] = k.eth_type
        """if k.src_mac:
            match_kwargs["eth_src"] = k.src_mac
        if k.dst_mac:
            match_kwargs["eth_dst"] = k.dst_mac"""
        if k.ip_src:
            match_kwargs["ipv4_src"] = k.ip_src
        if k.ip_dst:
            match_kwargs["ipv4_dst"] = k.ip_dst
        if k.ip_proto is not None:
            match_kwargs["ip_proto"] = int(k.ip_proto)
        # Clasificación + puerto estable (si aplica)
        cls, stable_side, stable_port, rule = self.classify_detail(desc, desc.dscp)
        # Si es CRIT y hay puerto estable, metemos SOLO ese puerto (no el efímero)
        prio = self.priority_base
        if cls == "CRIT" and stable_side and stable_port is not None and k.ip_proto in (6, 17):
            field = ("tcp_" if k.ip_proto == 6 else "udp_") + stable_side
            match_kwargs[field] = int(stable_port)
            prio = self.priority_base + 10  # CRIT gana a BE agregado
        
        # si es BE, hacer el match más "L3"
        """if cls == "BE":
            match_kwargs.pop("eth_src", None)
            match_kwargs.pop("eth_dst", None)"""


        queue_id = None
        if self.use_queues:
            queue_id = self.queue_id_crit if cls == "CRIT" else self.queue_id_be

        for i in range(len(path)):
            u = path[i]
            dp = self.topo.datapaths.get(u)
            if u == path[-1]:
                out_port = last_port
            else:
                v = path[i+1]
                if dp is None:
                    continue
                out_port = self.topo.neigh.get(u, {}).get(v)
                if out_port is None:
                    continue

            parser = dp.ofproto_parser
            ofp = dp.ofproto

            match = parser.OFPMatch(**match_kwargs)
            actions = []
            if queue_id is not None:
                actions.append(parser.OFPActionSetQueue(queue_id))
            actions.append(parser.OFPActionOutput(out_port))
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

            mod = parser.OFPFlowMod(
                datapath=dp,
                cookie=cookie,
                cookie_mask=0,
                table_id=table_id,
                command=ofp.OFPFC_ADD,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout,
                priority=prio,
                match=match,
                instructions=inst
            )
            dp.send_msg(mod)

        # Update indexes
        self.cookie_desc[cookie] = desc
        if cookie not in self.cookie_class:
            self.cookie_class[cookie] = cls

        self._register_cookie_path(cookie, path)
        k = desc.key
        if self.log_enabled:
            self.log.info(
                "[FLOW] install cookie=%s class=%s stable=%s:%s queue=%s src=%s:%s dst=%s:%s proto=%s path=%s",
                hex(cookie),
                self.cookie_class.get(cookie, "BE"),
                stable_side or "-", stable_port if stable_port is not None else "-",
                queue_id if queue_id is not None else "-",
                k.ip_src or "-", k.l4_src if k.l4_src is not None else "-",
                k.ip_dst or "-", k.l4_dst if k.l4_dst is not None else "-",
                k.ip_proto if k.ip_proto is not None else "-",
                path,
            )


    def delete_cookie_exact(self, cookie: int, dpids: Optional[List[int]] = None):
        """Borra reglas con cookie exacta en switches del path (o todos si dpids None)."""
        targets = dpids or list(self.topo.datapaths.keys())
        for dpid in targets:
            dp = self.topo.datapaths.get(dpid)
            if dp is None:
                continue
            parser = dp.ofproto_parser
            ofp = dp.ofproto
            mod = parser.OFPFlowMod(
                datapath=dp,
                table_id=ofp.OFPTT_ALL,
                command=ofp.OFPFC_DELETE,
                out_port=ofp.OFPP_ANY,
                out_group=ofp.OFPG_ANY,
                match=parser.OFPMatch(),
                cookie=cookie,
                cookie_mask=0xFFFFFFFFFFFFFFFF
            )
            dp.send_msg(mod)

    def reroute_cookie(self, old_cookie: int, new_path: List[int], table_id: int = 0) -> Optional[int]:
        """Make-before-break: instala versión nueva, barrier, borra versión vieja, actualiza índices."""
        desc = self.cookie_desc.get(old_cookie)
        dst_mac = desc.key.dst_mac if desc else None
        if desc is None:
            return None
        cls = self.cookie_class.get(old_cookie, "BE")
        rate = self.cookie_rate_mbps.get(old_cookie, 0.0)
        old_path = self.cookie_path.get(old_cookie, [])
        new_cookie = self.next_cookie(old_cookie)

        # install new
        self.install_path(desc, new_cookie, new_path, self.hosts[dst_mac][1], table_id=table_id)
        # barrier on all dps in new path
        if self.use_barrier:
            for dpid in set(new_path):
                dp = self.topo.datapaths.get(dpid)
                if dp:
                    self._send_barrier(dp)

        # delete old in old path switches
        old_path = self.cookie_path.get(old_cookie, [])
        self.delete_cookie_exact(old_cookie, dpids=old_path)
        if self.use_barrier:
            for dpid in set(old_path):
                dp = self.topo.datapaths.get(dpid)
                if dp:
                    self._send_barrier(dp)

        # move indexes
        last = self.cookie_last_move.get(old_cookie, 0.0)

        self._unregister_cookie(old_cookie)

        self.cookie_class[new_cookie] = cls
        self.cookie_rate_mbps[new_cookie] = rate
        self.cookie_last_move[new_cookie] = last
        cls2, s_side, s_port, _ = self.classify_detail(desc, desc.dscp)
        ckey = self._cookie_key(desc, cls2, s_side, s_port)
        self._active_cookie_by_key[ckey] = new_cookie


        if self.log_enabled:
            self.log.info(
                "[TE] reroute cookie=%s->%s class=%s rate=%.3f old_path=%s new_path=%s",
                hex(old_cookie),
                hex(new_cookie),
                cls,
                rate,
                old_path,
                new_path,
            )

        return new_cookie

    def learn_host(self, dpid: int, src_mac: str, in_port: int):
        if src_mac not in self.hosts:
            self.hosts[src_mac] = (dpid, in_port)

    # ---------- Helper to create FlowDescriptor from parsed packet ----------
    @staticmethod
    def build_flow_key(eth, ip4, l4) -> FlowKey:
        eth_type = eth.ethertype
        src_mac = eth.src
        dst_mac = eth.dst
        if ip4 is None:
            return FlowKey(eth_type, src_mac, dst_mac, None, None, None, None, None)
        ip_src = ip4.src
        ip_dst = ip4.dst
        ip_proto = int(ip4.proto)
        l4_src = None
        l4_dst = None
        if l4 is not None:
            if hasattr(l4, "src_port"):
                l4_src = int(l4.src_port)
            if hasattr(l4, "dst_port"):
                l4_dst = int(l4.dst_port)
        return FlowKey(eth_type, src_mac, dst_mac, ip_src, ip_dst, ip_proto, l4_src, l4_dst)
