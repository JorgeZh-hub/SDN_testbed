# topology_manager.py
from collections import defaultdict, deque
from ryu.ofproto import ofproto_v1_3

class TopologyManager:
    def __init__(self, logger):
        self.log = logger
        self.neigh = defaultdict(dict)      # neigh[u][v] = port u->v
        self.switches = []
        self.datapath_list = {}             # dpid -> dp
        self.all_ports = defaultdict(set)   # dpid -> ports
        self.lldp_ports = defaultdict(set)  # dpid -> lldp seen ports

        self.tree_ports = defaultdict(set)  # dpid -> inter-switch ports in spanning tree
        self.tree_root = None

    def on_switch_enter(self, dp):
        dpid = dp.id
        if dpid not in self.switches:
            self.switches.append(dpid)
            self.datapath_list[dpid] = dp
            self.all_ports[dpid] = set(dp.ports.keys())

            self.log.info(f"[SWITCH ENTER] dpid={dpid} ports={sorted(self.all_ports[dpid])}")

        self.recompute_spanning_tree()
        self.log.info(f"[STP] root={self.tree_root} tree_ports={ {k: sorted(list(v)) for k,v in self.tree_ports.items()} }")


    def on_switch_leave(self, dpid):
        if dpid in self.switches:
            self.switches.remove(dpid)
        self.datapath_list.pop(dpid, None)
        self.neigh.pop(dpid, None)
        self.recompute_spanning_tree()

    def on_link_add(self, src_dpid, src_port, dst_dpid, dst_port):
        self.neigh[src_dpid][dst_dpid] = src_port
        self.neigh[dst_dpid][src_dpid] = dst_port
        self.log.info(f"[LINK ADD] {src_dpid}:{src_port} <-> {dst_dpid}:{dst_port}")
        self.recompute_spanning_tree()


    def on_link_delete(self, s1, s2):
        self.log.warning(f"[LINK DOWN] {s1} <-> {s2}")
        try:
            del self.neigh[s1][s2]
            del self.neigh[s2][s1]
        except KeyError:
            pass
        self.recompute_spanning_tree()


    def recompute_spanning_tree(self):
        self.tree_ports.clear()
        if not self.switches:
            self.tree_root = None
            return

        root = min(self.switches)
        self.tree_root = root
        visited = {root}
        q = deque([root])

        while q:
            u = q.popleft()
            neighbors = sorted(self.neigh[u].keys())
            for v in neighbors:
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

    def trunk_ports(self, dpid):
        # trunks conocidos por vecindad + LLDP
        return set(self.neigh.get(dpid, {}).values()) | self.lldp_ports.get(dpid, set())

    def flood_ports_on_tree(self, dpid, in_port):
        trunk = self.trunk_ports(dpid)
        edge_ports = self.all_ports.get(dpid, set()) - trunk
        ports = (self.tree_ports.get(dpid, set()) | edge_ports)
        ports.discard(in_port)
        ports.discard(ofproto_v1_3.OFPP_LOCAL)
        return sorted(ports)
