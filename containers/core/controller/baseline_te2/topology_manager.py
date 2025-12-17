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
        # directed src->dst
        self.neigh[src_dpid][dst_dpid] = src_port
        self.port_to_neigh[src_dpid][src_port] = dst_dpid
        self.link_ports[(src_dpid, dst_dpid)] = (src_port, dst_port)

        # recompute tree (cheap BFS)
        self._recompute_spanning_tree()

    def del_link(self, src_dpid: int, dst_dpid: int):
        self.neigh.get(src_dpid, {}).pop(dst_dpid, None)
        # remove port_to_neigh entry
        for p,v in list(self.port_to_neigh.get(src_dpid, {}).items()):
            if v == dst_dpid:
                self.port_to_neigh[src_dpid].pop(p, None)
        self.link_ports.pop((src_dpid, dst_dpid), None)
        self._recompute_spanning_tree()

    def _recompute_spanning_tree(self):
        # Simple BFS from smallest dpid to avoid L2 loops on flood
        dpids = sorted(self.datapaths.keys())
        if not dpids:
            self.tree_ports.clear()
            self.tree_root = None
            return
        root = dpids[0]
        self.tree_root = root
        self.tree_ports = defaultdict(set)

        visited = set([root])
        q = deque([root])
        while q:
            u = q.popleft()
            for v, outp in self.neigh.get(u, {}).items():
                if v in visited:
                    continue
                # u->v is in tree
                self.tree_ports[u].add(outp)
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

    def lookup_mac_port(self, dpid: int, dst_mac: str) -> Optional[int]:
        return self.mac_to_port.get(dpid, {}).get(dst_mac)
