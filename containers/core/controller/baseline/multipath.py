#!/usr/bin/python3

from containers.core.controller.baseline.import_multipath import *

REFERENCE_BW = 10000000
DEFAULT_BW = 10000000
MAX_PATHS = 1
FLOW_PRIORITY = 100
APP_ID = 0x13
COOKIE_TTL = 120   # segundos sin uso
GC_INTERVAL = 10   # cada cuánto revisar


@dataclass
class Paths:
    ''' Paths container'''
    path: None
    cost: float

class Controller13(app_manager.RyuApp): 
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Controller13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}    
        self.neigh = defaultdict(dict)                                  # Puertos conectados entre dos switches
        self.bw = defaultdict(lambda: defaultdict( lambda: DEFAULT_BW)) # Anchos de banda entre cada link, por default DEFAULT_BW
        self.prev_bytes = defaultdict(lambda: defaultdict( lambda: 0))  # Bytes transmitidos desde el último OFPPortStatsRequest, usado para calcular el ancho de banda percibido en un puerto
        self.hosts = {}                                                 # (Switch id, Puerto) al que están conectados los hosts (por MAC)                                   
        self.switches = []                                              # ID de switches activos
        self.arp_table = {}                                             # IP y MACs de los hosts                              
        self.path_table = {}                                            # (Id switch inicio, puerto de ingreso, id switch final, puerto de ingreso) para cada ruta óptima
        self.paths_table = {}                                           # Todas las rutas descubiertas en la red
        self.path_with_ports_table = {}                                 # SwitchIDs con los puertos de entrada y salida para las rutas
        self.datapath_list = {}                                         # IDs de switches origen
        self.monitor_thread = hub.spawn(self._monitor)
        self.tree_ports = defaultdict(set)     # dpid -> puertos inter-switch que pertenecen al spanning tree
        self.tree_root = None
        self.all_ports = defaultdict(set)   # dpid -> todos los puertos del switch
        self.lldp_ports = defaultdict(set)   # dpid -> puertos donde vi LLDP (trunks reales)
        self.cookie_installed = defaultdict(set)   # cookie -> {dpid1, dpid2, ...}
        self.cookie_path_nodes = {}                # cookie -> [dpid en orden del path]
        self.cookie_last_seen = {}                 # cookie -> timestamp (para GC)
        self.gc_thread = hub.spawn(self._cookie_gc)
    def _cookie_gc(self):
        while True:
            now = time.time()
            to_delete = [c for c, last in list(self.cookie_last_seen.items())
                        if (now - last) > COOKIE_TTL]
            for c in to_delete:
                self.delete_cookie_everywhere(c)
            hub.sleep(GC_INTERVAL)


    def make_cookie(self, flow_type, ip_src, ip_dst):
        base = f"{flow_type}|{ip_src}|{ip_dst}"
        h = zlib.crc32(base.encode()) & 0xFFFFFFFF
        return (APP_ID << 48) | h

    def delete_flows_by_cookie(self, dp, cookie):
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        mod = parser.OFPFlowMod(
            datapath=dp,
            command=ofp.OFPFC_DELETE,
            out_port=ofp.OFPP_ANY,
            out_group=ofp.OFPG_ANY,
            match=parser.OFPMatch(),
            cookie=cookie,
            cookie_mask=0xFFFFFFFFFFFFFFFF
        )
        dp.send_msg(mod)

    def delete_cookie_everywhere(self, cookie):
        for dpid in list(self.cookie_installed.get(cookie, set())):
            dp = self.datapath_list.get(dpid)
            if dp:
                self.delete_flows_by_cookie(dp, cookie)

        self.cookie_installed.pop(cookie, None)
        self.cookie_path_nodes.pop(cookie, None)
        self.cookie_last_seen.pop(cookie, None)




    def recompute_spanning_tree(self):
        """Calcula un spanning tree global usando BFS desde el switch con DPID menor."""
        self.tree_ports.clear()

        if not self.switches:
            self.tree_root = None
            return

        root = min(self.switches)
        self.tree_root = root

        visited = set([root])
        q = deque([root])

        while q:
            u = q.popleft()

            # vecinos en orden estable
            neighbors = list(self.neigh[u].keys())
            neighbors.sort()

            for v in neighbors:
                if v in visited:
                    continue

                # agrega arista u-v al árbol: guarda puertos de salida en ambos switches
                try:
                    puv = self.neigh[u][v]
                    pvu = self.neigh[v][u]
                except KeyError:
                    continue

                self.tree_ports[u].add(puv)
                self.tree_ports[v].add(pvu)

                visited.add(v)
                q.append(v)

    def flood_ports_on_tree(self, dpid, in_port, trunk_ports):
        edge_ports = self.all_ports.get(dpid, set()) - trunk_ports
        ports = (self.tree_ports.get(dpid, set()) | edge_ports)
        ports.discard(in_port)
        ports.discard(ofproto_v1_3.OFPP_LOCAL)  # o 0 si tu setup lo usa así
        return sorted(ports)


    def _monitor(self):
        while True:
            for dp in self.datapath_list.values():
                self._request_stats(dp)
            hub.sleep(1)

    # Petición de estadísticas
    def _request_stats(self, dp):
        ofp_parser = dp.ofproto_parser
        dp.send_msg(ofp_parser.OFPPortStatsRequest(dp))
        dp.send_msg(ofp_parser.OFPFlowStatsRequest(dp))
    
    def _path_uses_link(self, nodes, s1, s2):
        return any(
            (nodes[i] == s1 and nodes[i+1] == s2) or (nodes[i] == s2 and nodes[i+1] == s1)
            for i in range(len(nodes) - 1)
        )

    def get_bandwidth(self, path, port, index):
    	return self.bw[path[index]][port]

    def find_path_cost(self, path):
        ''' arg path is a list with all nodes in our route '''
        path_cost = []
        i = 0
        while(i < len(path) - 1):
            port1 = self.neigh[path[i]][path[i + 1]]
            bandwidth_between_two_nodes = self.get_bandwidth(path, port1, i)
            path_cost.append(bandwidth_between_two_nodes)
            i += 1
        return sum(path_cost)


    def find_paths_and_costs(self, src, dst):
        ''' 
        Implementation of Breath-First Search Algorithm (BFS) 
        Output of this function returns an list on class Paths objects
        ''' 
        if src == dst:
            return [Paths([src], 0)]

        q = deque([(src, [src])])
        possible_paths = []

        while q:
            node, path = q.popleft()
            visited = set(path)

            neighs = list(self.neigh[node].keys())
            neighs.sort()

            for v in neighs:
                if v in visited:
                    continue

                new_path = path + [v]
                if v == dst:
                    cost = self.find_path_cost(new_path)
                    possible_paths.append(Paths(new_path, cost))
                else:
                    q.append((v, new_path))

        return possible_paths
           
    def find_n_optimal_paths(self, paths, number_of_optimal_paths=MAX_PATHS):
        k = number_of_optimal_paths
        return sorted(paths, key=lambda p: (p.cost, len(p.path), tuple(p.path)))[:k]
    
    def add_ports_to_paths(self, paths, first_port, last_port):
        '''
        Add the ports to all switches including hosts
        '''
        paths_n_ports = list()
        bar = dict()
        in_port = first_port
        for s1, s2 in zip(paths[0].path[:-1], paths[0].path[1:]):
            out_port = self.neigh[s1][s2]                          # Puerto de salida s1 conectado a s2
            bar[s1] = (in_port, out_port)                          # Puerto de entrada y salida en el switch
            in_port = self.neigh[s2][s1]                           # Puerto de entrada s2 conectado a s1
        bar[paths[0].path[-1]] = (in_port, last_port)              # Puerto de entrada y salida en el switch final
        paths_n_ports.append(bar)
        return paths_n_ports

    def install_paths(self, src, first_port, dst, last_port, ip_src, ip_dst, type, pkt):
        cookie = self.make_cookie(type, ip_src, ip_dst)
        self.cookie_last_seen[cookie] = time.time()

        key = (src, first_port, dst, last_port)
        if key not in self.path_table or key not in self.path_with_ports_table:
            self.topology_discover(src, first_port, dst, last_port)
            self.topology_discover(dst, last_port, src, first_port)

        nodes_in_path = self.path_table[key][0].path
        self.cookie_path_nodes[cookie] = nodes_in_path


        for node in nodes_in_path:
            # ✅ ya instalé este “flujo general” en este switch
            if node in self.cookie_installed[cookie]:
                continue

            dp = self.datapath_list[node]
            ofp_parser = dp.ofproto_parser

            in_port = self.path_with_ports_table[key][0][node][0]
            out_port = self.path_with_ports_table[key][0][node][1]
            actions = [ofp_parser.OFPActionOutput(out_port)]

            if type == 'IP':
                match = ofp_parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=ip_src,
                    ipv4_dst=ip_dst
                )
                self.add_flow(dp, FLOW_PRIORITY, match, actions, idle_timeout=0, cookie=cookie)

            elif type == 'ARP':
                match_arp = ofp_parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_ARP,
                    arp_spa=ip_src,
                    arp_tpa=ip_dst
                )
                self.add_flow(dp, FLOW_PRIORITY, match_arp, actions, idle_timeout=0, cookie=cookie)
            self.cookie_installed[cookie].add(node)


        return self.path_with_ports_table[key][0][src][1]

    def add_flow(self, datapath, priority, match, actions, idle_timeout, buffer_id=None, cookie=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id is not None and buffer_id != ofproto.OFP_NO_BUFFER:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                idle_timeout=idle_timeout,
                hard_timeout=0,
                cookie=cookie,
                cookie_mask=0,
                instructions=inst
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                idle_timeout=idle_timeout,
                hard_timeout=0,
                cookie=cookie,
                cookie_mask=0,
                instructions=inst
            )

        datapath.send_msg(mod)
        return True


    def topology_discover(self, src, first_port, dst, last_port):
        paths = self.find_paths_and_costs(src, dst)
        path = self.find_n_optimal_paths(paths)
        path_with_port = self.add_ports_to_paths(path, first_port, last_port)           # SwitchIDs con los puertos de entrada y salida para la ruta
        
        self.logger.info(f"Possible paths: {paths}")
        self.logger.info(f"Optimal Path with port: {path_with_port}")
        
        self.paths_table[(src, first_port, dst, last_port)]  = paths
        self.path_table[(src, first_port, dst, last_port)] = path
        self.path_with_ports_table[(src, first_port, dst, last_port)] = path_with_port

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes", ev.msg.msg_len, ev.msg.total_len)

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]   
        dpid = datapath.id

        # Ignorar LLDP e IPV6

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            self.lldp_ports[dpid].add(in_port)
            return
        if eth.ethertype == 0x86dd:
            return

        dst = eth.dst
        src = eth.src

        trunk_ports = set(self.neigh.get(dpid, {}).values()) | self.lldp_ports.get(dpid, set())
        flood_ports = self.flood_ports_on_tree(dpid, in_port, trunk_ports)

        if in_port not in trunk_ports and src not in self.hosts:
            self.hosts[src] = (dpid, in_port)

        out_port = None

        if eth.ethertype == ether_types.ETH_TYPE_IP:
            nw = pkt.get_protocol(ipv4.ipv4)
            if nw is None:
                return
            src_ip, dst_ip = nw.src, nw.dst
            self.arp_table[src_ip] = src

            if dst in self.hosts and src in self.hosts:
                h1 = self.hosts[src]
                h2 = self.hosts[dst]
                self.logger.info(f"IP from: {src_ip} to: {dst_ip}")
                out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip, 'IP', pkt)
                self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip, 'IP', pkt)

        elif eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt is None:
                return

            src_ip = arp_pkt.src_ip
            dst_ip = arp_pkt.dst_ip
            key = (src_ip, dst_ip, in_port)
            now = time.time()

            if arp_pkt.opcode == arp.ARP_REPLY:
                self.arp_table[src_ip] = src
                if dst in self.hosts and src in self.hosts:
                    h1 = self.hosts[src]
                    h2 = self.hosts[dst]
                    self.logger.info(f"ARP Reply from: {src_ip} to: {dst_ip} H1: {h1} H2: {h2}")
                    out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip, 'ARP', pkt)
                    self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip, 'ARP', pkt)

            elif arp_pkt.opcode == arp.ARP_REQUEST:
                self.arp_table[src_ip] = src
                self.logger.info(f"ARP Request procesado de: {src_ip} → {dst_ip} | Switch {dpid} puerto {in_port}")

        if out_port is None:
            if flood_ports:
                actions = [parser.OFPActionOutput(p) for p in flood_ports]
            else:
                self.logger.info(f"[DROP] flood_ports vacío en sw {dpid}, in_port {in_port}, eth_type {hex(eth.ethertype)}")
                return
        else:
            actions = [parser.OFPActionOutput(out_port)]

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # 🔁 1. Regla para descartar IPv6
        match_ipv6 = parser.OFPMatch(eth_type=0x86dd)
        self.add_flow(datapath, priority=FLOW_PRIORITY, match=match_ipv6, actions=[], idle_timeout=0)

        # ⚙️ 2. Regla por defecto: enviar al controlador
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, idle_timeout=0)


    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        '''Reply to the OFPPortStatsRequest, visible beneath'''
        switch_dpid = ev.msg.datapath.id
        for p in ev.msg.body:
            self.bw[switch_dpid][p.port_no] = (p.tx_bytes - self.prev_bytes[switch_dpid][p.port_no])*8.0/1000000     # Calcula el ancho de banda en función del número de bytes transmitidos desde el último OFPPortStatsRequest
            self.prev_bytes[switch_dpid][p.port_no] = p.tx_bytes                                                     # Almacena los bytes transmitidos en la última solicitud OFPPortStatsRequest

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        switch_dp = ev.switch.dp
        switch_dpid = switch_dp.id
        ofp_parser = switch_dp.ofproto_parser
        
        self.logger.info(f"Switch has been plugged in PID: {switch_dpid}")
            
        if switch_dpid not in self.switches:
            self.datapath_list[switch_dpid] = switch_dp
            self.switches.append(switch_dpid)
            self.all_ports[switch_dp.id] = set(switch_dp.ports.keys())  # puertos conocidos por el datapath

        self.cookie_installed.clear()
        self.cookie_path_nodes.clear()
        self.cookie_last_seen.clear()
        self.recompute_spanning_tree()

    @set_ev_cls(event.EventSwitchLeave, MAIN_DISPATCHER)
    def switch_leave_handler(self, ev):
        switch = ev.switch.dp.id
        if switch in self.switches:
            try:
                self.switches.remove(switch)
                del self.datapath_list[switch]
                del self.neigh[switch]
            except KeyError:
                self.logger.info(f"Switch has been already pulged off PID{switch}!")
        self.recompute_spanning_tree()
            

    @set_ev_cls(event.EventLinkAdd, MAIN_DISPATCHER)
    def link_add_handler(self, ev):
        self.neigh[ev.link.src.dpid][ev.link.dst.dpid] = ev.link.src.port_no
        self.neigh[ev.link.dst.dpid][ev.link.src.dpid] = ev.link.dst.port_no
        self.logger.info(
            f"Link established: {ev.link.src.dpid}:{ev.link.src.port_no} <-> "
            f"{ev.link.dst.dpid}:{ev.link.dst.port_no}"
        )
        self.recompute_spanning_tree()


    @set_ev_cls(event.EventLinkDelete, MAIN_DISPATCHER)
    def link_delete_handler(self, ev):
        s1, s2 = ev.link.src.dpid, ev.link.dst.dpid
        self.logger.warning(f"Link down: {s1} <-> {s2}")

        try:
            del self.neigh[s1][s2]
            del self.neigh[s2][s1]
        except KeyError:
            self.logger.info("Link ya eliminado")
            return

        affected_cookies = [c for c, nodes in list(self.cookie_path_nodes.items())
                    if self._path_uses_link(nodes, s1, s2)]

        for c in affected_cookies:
            self.delete_cookie_everywhere(c)

        afectados = [key for key, path_list in self.path_table.items()
             if self._path_uses_link(path_list[0].path, s1, s2)]
        
        for key in afectados:
            self.path_table.pop(key, None)
            self.path_with_ports_table.pop(key, None)
            self.paths_table.pop(key, None)
            src_sw, src_port, dst_sw, dst_port = key
            self.topology_discover(src_sw, src_port, dst_sw, dst_port)
        self.recompute_spanning_tree()

