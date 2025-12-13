#!/usr/bin/python3

from import_multipath import *

REFERENCE_BW = 10000000
DEFAULT_BW = 10000000
MAX_PATHS = 1
FLOW_PRIORITY = 100

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
        self.path_calculation_keeper = []                               # Rutas ya instaladas
        self.arp_cache = defaultdict(dict)                              # Paquetes ARP procesados en el switch para evitar flooding
        self.arp_request_counter = 0
        self.arp_cache_timeout = 3  # tiempo en segundos
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
    # Bucle cooperativo único
    def _monitor(self):
        while True:
            for dp in self.datapath_list.values():
                self._request_stats(dp)
            hub.sleep(1)          # intervalo en segundos

    # Petición de estadísticas
    def _request_stats(self, dp):
        ofp_parser = dp.ofproto_parser
        dp.send_msg(ofp_parser.OFPPortStatsRequest(dp))
        dp.send_msg(ofp_parser.OFPFlowStatsRequest(dp))

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
            return [Paths(src,0)]
        queue = [(src, [src])]
        possible_paths = list() 
        while queue:
            (edge, path) = queue.pop()
            for vertex in set(self.neigh[edge]) - set(path):
                if vertex == dst:
                    path_to_dst = path + [vertex]
                    cost_of_path = self.find_path_cost(path_to_dst)
                    possible_paths.append(Paths(path_to_dst, cost_of_path))
                else:
                    queue.append((vertex, path + [vertex]))
        return possible_paths 
           
    def find_n_optimal_paths(self, paths, number_of_optimal_paths = MAX_PATHS):
        '''arg paths is an list containing lists of possible paths'''
        costs = [path.cost for path in paths]
        optimal_paths_indexes = list(map(costs.index, heapq.nsmallest(number_of_optimal_paths,costs)))
        optimal_paths = [paths[op_index] for op_index in optimal_paths_indexes]
        return optimal_paths
    
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
        # 🔒 Verifica si la ruta ya está calculada y activa
        #if (src, first_port, dst, last_port) in self.path_table:
         #   self.logger.info(f"Ruta ya existente entre {src}->{dst}, no se recalcula.")
          #  return self.path_with_ports_table[(src, first_port, dst, last_port)][0][src][1]
        
        # 💡 Si no existe, se instala por primera vez
        if (src, first_port, dst, last_port) not in self.path_calculation_keeper:
            self.path_calculation_keeper.append((src, first_port, dst, last_port))
            self.topology_discover(src, first_port, dst, last_port)
            self.topology_discover(dst, last_port, src, first_port)


        for node in self.path_table[(src, first_port, dst, last_port)][0].path:

            dp = self.datapath_list[node]
            ofp = dp.ofproto
            ofp_parser = dp.ofproto_parser

            actions = []

            in_port = self.path_with_ports_table[(src, first_port, dst, last_port)][0][node][0]
            out_port = self.path_with_ports_table[(src, first_port, dst, last_port)][0][node][1]
                
            actions = [ofp_parser.OFPActionOutput(out_port)]

            if type == 'IP':                      # nuevo caso único para IP
                match = ofp_parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=ip_src,
                    ipv4_dst=ip_dst)
                self.logger.info(f"Path installed in sw {node}: {in_port}→{out_port}")
                self.add_flow(dp, FLOW_PRIORITY, match, actions, idle_timeout=0)
            elif type == 'ARP':
                match_arp = ofp_parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_ARP,
                                 arp_spa=ip_src, arp_tpa=ip_dst)
                self.add_flow(dp, priority=FLOW_PRIORITY, match=match_arp, actions=actions, idle_timeout=0)
                self.logger.info(f"Install path in switch: {node} out port: {out_port} in port: {in_port} ")
                self.logger.info("ARP Flow added ! ")
        
        return self.path_with_ports_table[(src, first_port, dst, last_port)][0][src][1]

    def add_flow(self, datapath, priority, match, actions, idle_timeout, buffer_id = None):
        ''' Method Provided by the source Ryu library.'''
        
        ofproto = datapath.ofproto 
        parser = datapath.ofproto_parser 

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, idle_timeout = idle_timeout, hard_timeout=0,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, idle_timeout = idle_timeout, hard_timeout=0, instructions=inst)
        datapath.send_msg(mod)

    def topology_discover(self, src, first_port, dst, last_port):
        #threading.Timer(1.0, self.topology_discover, args=(src, first_port, dst, last_port)).start()
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
        arp_pkt = pkt.get_protocol(arp.arp)

        
        # Ignorar LLDP e IPV6
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:# or eth.ethertype == 0x86dd:
            return 
        
        if eth.ethertype == 0x86dd:
            print("Paquete IPV6 detectado.... Revisar")

        print("---------------------------Nuevo packetIN---------------------------------------")

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        if src not in self.hosts:
            self.hosts[src] = (dpid, in_port)                    # Almacenamiento del host que está conectado al switch dpid en el puerto in_port

        out_port = ofproto.OFPP_FLOOD                            # Flooding de OpenFlow, por defecto

        if eth.ethertype == ether_types.ETH_TYPE_IP:
            nw = pkt.get_protocol(ipv4.ipv4)
            src_ip, dst_ip = nw.src, nw.dst
            self.arp_table[src_ip] = src
            h1 = self.hosts[src]
            h2 = self.hosts[dst]

            self.logger.info(f"IP from: {src_ip} to: {dst_ip}")
            out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip, 'IP', pkt)
            self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip, 'IP', pkt)
        elif eth.ethertype == ether_types.ETH_TYPE_ARP:
                src_ip = arp_pkt.src_ip
                dst_ip = arp_pkt.dst_ip
                key = (src_ip, dst_ip)
                now = time.time()

                if arp_pkt.opcode == arp.ARP_REPLY:
                    self.arp_table[src_ip] = src
                    h1 = self.hosts[src]
                    h2 = self.hosts[dst]

                    self.logger.info(f" ARP Reply from: {src_ip} to: {dst_ip} H1: {h1} H2: {h2}")
                    out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip, 'ARP', pkt)
                    self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip, 'ARP', pkt)

                elif arp_pkt.opcode == arp.ARP_REQUEST:
                    self.arp_table[src_ip] = src

                    # Filtro por tiempo (evita repetir en menos de N segundos)
                    if key in self.arp_cache[dpid]:
                        last_time = self.arp_cache[dpid][key]
                        if now - last_time < self.arp_cache_timeout:
                            self.logger.info(f"ARP Request ignorado por timeout: {src_ip} → {dst_ip}, hace %.2fs", now - last_time)
                            return

                    # Si no está o ya expiró
                    self.arp_cache[dpid][key] = now
                    self.logger.info(f"ARP Request procesado de: {src_ip} → {dst_ip} | Switch {dpid} puerto {in_port}")

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
            

    @set_ev_cls(event.EventLinkAdd, MAIN_DISPATCHER)
    def link_add_handler(self, ev):
        self.neigh[ev.link.src.dpid][ev.link.dst.dpid] = ev.link.src.port_no
        self.neigh[ev.link.dst.dpid][ev.link.src.dpid] = ev.link.dst.port_no
        self.logger.info(f"Link between switches has been established, SW1 DPID: {ev.link.src.dpid}:{ev.link.dst.port_no} SW2 DPID: {ev.link.dst.dpid}:{ev.link.dst.port_no}")

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

        afectados = [key for key, path in self.path_table.items()
                    if s1 in path[0].path and s2 in path[0].path]

        for (src_sw, src_port, dst_sw, dst_port) in afectados:
            del self.path_table[(src_sw, src_port, dst_sw, dst_port)]
            del self.path_with_ports_table[(src_sw, src_port, dst_sw, dst_port)]
            self.topology_discover(src_sw, src_port, dst_sw, dst_port)
