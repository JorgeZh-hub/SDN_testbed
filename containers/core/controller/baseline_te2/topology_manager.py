# topology_manager.py
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Tuple, Optional, Set, List

from ryu.ofproto import ofproto_v1_3

class TopologyManager:
    """Mantiene grafo L2 (solo switches) + mapeo puertos/nombres.

    neigh[u][v] = out_port en u hacia v (link dirigido)
    port_to_neigh[u][out_port] = v
    port_name[(dpid, port_no)] = "s1-eth1" o "switch1-eth1" (según OVS)
    """

    def __init__(self, logger):
        self.log = logger

        self.datapaths: Dict[int, object] = {}
        self.neigh: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.port_to_neigh: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.link_ports: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (u,v)->(u_port,v_port)

        self.all_ports: Dict[int, Set[int]] = defaultdict(set)
        self.port_name: Dict[Tuple[int, int], str] = {}

        # for flood without loops
        self.tree_ports: Dict[int, Set[int]] = defaultdict(set)
        self.tree_root: Optional[int] = None

        # MAC learning: mac -> (dpid,port)
        self.mac_to_port: Dict[int, Dict[str, int]] = defaultdict(dict)
        # Host location (first seen): mac -> (dpid, port)
        self.hosts: Dict[str, Tuple[int, int]] = {}

    # ---------------- Datapaths & Ports ----------------
    def register_dp(self, dp):
        self.datapaths[dp.id] = dp

    def unregister_dp(self, dpid: int):
        self.datapaths.pop(dpid, None)
        self.neigh.pop(dpid, None)
        self.port_to_neigh.pop(dpid, None)
        self.all_ports.pop(dpid, None)
        # remove link entries involving dpid
        for (u,v) in list(self.link_ports.keys()):
            if u == dpid or v == dpid:
                self.link_ports.pop((u,v), None)

    def update_port_desc(self, dpid: int, ports: List[object]):
        # ports are OFPPort objects
        for p in ports:
            self.all_ports[dpid].add(p.port_no)
            if hasattr(p, "name") and p.name:
                try:
                    nm = p.name.decode() if isinstance(p.name, (bytes, bytearray)) else str(p.name)
                except Exception:
                    nm = str(p.name)
                self.port_name[(dpid, p.port_no)] = nm

    def get_port_name(self, dpid: int, port_no: int) -> Optional[str]:
        return self.port_name.get((dpid, port_no))

    
    # ---------------- Links ----------------
    def add_link(self, src_dpid: int, src_port: int, dst_dpid: int, dst_port: int):
        self.neigh[src_dpid][dst_dpid] = src_port
        self.neigh[dst_dpid][src_dpid] = dst_port

        self.port_to_neigh[src_dpid][src_port] = dst_dpid
        self.port_to_neigh[dst_dpid][dst_port] = src_dpid

        self.link_ports[(src_dpid, dst_dpid)] = (src_port, dst_port)
        self.link_ports[(dst_dpid, src_dpid)] = (dst_port, src_port)

        self._recompute_spanning_tree()

    def del_link(self, dpid1: int, dpid2: int):
        self.neigh.get(dpid1, {}).pop(dpid2, None)
        self.neigh.get(dpid2, {}).pop(dpid1, None)

        # limpia port_to_neigh en ambos switches
        for p, v in list(self.port_to_neigh.get(dpid1, {}).items()):
            if v == dpid2:
                self.port_to_neigh[dpid1].pop(p, None)
        for p, v in list(self.port_to_neigh.get(dpid2, {}).items()):
            if v == dpid1:
                self.port_to_neigh[dpid2].pop(p, None)

        self.link_ports.pop((dpid1, dpid2), None)
        self.link_ports.pop((dpid2, dpid1), None)

        self._recompute_spanning_tree()

    def _recompute_spanning_tree(self):
        dpids = sorted(self.datapaths.keys())
        self.tree_ports = defaultdict(set)

        if not dpids:
            self.tree_root = None
            return

        root = dpids[0]
        self.tree_root = root

        visited = {root}
        q = deque([root])

        while q:
            u = q.popleft()
            for v in sorted(self.neigh.get(u, {}).keys()):
                if v in visited:
                    continue
                try:
                    puv = self.neigh[u][v] 
                    pvu = self.neigh[v][u] 
                except KeyError:
                    continue

                self.tree_ports[u].add(puv)
                self.tree_ports[v].add(pvu)

                visited.add(v)
                q.append(v)

    def trunk_ports(self, dpid: int) -> Set[int]:
        # ports that go to other switches
        return set(self.neigh.get(dpid, {}).values())

    def flood_ports_on_tree(self, dpid: int, in_port: int) -> List[int]:
        trunk = self.trunk_ports(dpid)
        edge_ports = self.all_ports.get(dpid, set()) - trunk
        ports = set(self.tree_ports.get(dpid, set())) | set(edge_ports)
        ports.discard(in_port)
        ports.discard(ofproto_v1_3.OFPP_LOCAL)
        return sorted(ports)

    # ---------------- L2 learning helpers ----------------
    def learn_mac(self, dpid: int, src_mac: str, in_port: int):
        self.mac_to_port[dpid][src_mac] = in_port

    def learn_host(self, dpid: int, src_mac: str, in_port: int):
        if src_mac not in self.hosts:
            self.hosts[src_mac] = (dpid, in_port)

    def lookup_mac_port(self, dpid: int, dst_mac: str) -> Optional[int]:
        return self.mac_to_port.get(dpid, {}).get(dst_mac)
