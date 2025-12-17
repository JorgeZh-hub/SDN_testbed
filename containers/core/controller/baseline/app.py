#!/usr/bin/python3
from imports import *

REFERENCE_BW = 10000000
DEFAULT_BW = 10000000
MAX_PATHS = 1
FLOW_PRIORITY = 100

APP_ID = 0x13
COOKIE_TTL = 120
GC_INTERVAL = 10

class Controller13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Controller13, self).__init__(*args, **kwargs)

        # --- estado en app (por tiempo) ---
        self.hosts = {}  # mac -> (dpid, port)
        self.arp_table = {}  # ip -> mac

        # stats BW 
        self.bw = defaultdict(lambda: defaultdict(lambda: DEFAULT_BW))
        self.prev_bytes = defaultdict(lambda: defaultdict(lambda: 0))
        self.monitor_thread = hub.spawn(self._monitor)

        # --- managers ---
        self.topo = TopologyManager(self.logger)

        # --- logers ---
        self._last_pkt_log = 0
        self.PKT_LOG_INTERVAL = 2  # segundos
        self._cookie_logged = set()


        self.flows = FlowManager(
            logger=self.logger,
            get_dp=lambda dpid: self.topo.datapath_list.get(dpid),
            app_id=APP_ID,
            cookie_ttl=COOKIE_TTL,
            gc_interval=GC_INTERVAL
        )

        self.paths = PathEngine(
            logger=self.logger,
            topo=self.topo,
            bw_getter=lambda u, port: self.bw[u][port],
            default_bw=DEFAULT_BW,
            max_paths=MAX_PATHS
        )

    # -------------------- STATS --------------------
    def _monitor(self):
        while True:
            for dp in self.topo.datapath_list.values():
                self._request_stats(dp)
            hub.sleep(1)

    def _request_stats(self, dp):
        parser = dp.ofproto_parser
        dp.send_msg(parser.OFPPortStatsRequest(dp))
        # Si no lo usas, mejor comentar:
        # dp.send_msg(parser.OFPFlowStatsRequest(dp))

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        for p in ev.msg.body:
            # “rate” aproximado (mantienes tu lógica)
            self.bw[dpid][p.port_no] = (p.tx_bytes - self.prev_bytes[dpid][p.port_no]) * 8.0 / 1000000
            self.prev_bytes[dpid][p.port_no] = p.tx_bytes

    # -------------------- FLOW INSTALL --------------------
    def install_paths(self, src, first_port, dst, last_port, ip_src, ip_dst, flow_type):
        """
        flow_type: 'IP' o 'ARP'
        Devuelve el out_port en el switch src para enviar el PacketOut inicial.
        """
        # asegura que el path exista (on-demand)
        key = self.paths.ensure_path(src, first_port, dst, last_port)

        nodes_in_path = self.paths.path_table[key][0].path
        ports_map = self.paths.path_with_ports_table[key][0]


        cookie = self.flows.make_cookie(flow_type, ip_src, ip_dst)
        self.flows.touch(cookie, nodes_in_path)

        if cookie not in self._cookie_logged:
            self._cookie_logged.add(cookie)
            self.logger.info(
                f"[PATH] cookie={hex(cookie)} type={flow_type} {ip_src}->{ip_dst} "
                f"path={nodes_in_path}"
            )

        for node in nodes_in_path:
            dp = self.topo.datapath_list[node]
            parser = dp.ofproto_parser

            in_port, out_port = ports_map[node]
            actions = [parser.OFPActionOutput(out_port)]

            if flow_type == 'IP':
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=ip_src,
                    ipv4_dst=ip_dst
                )
                self.flows.install_once(dp, node, cookie, FLOW_PRIORITY, match, actions, idle_timeout=0)

            elif flow_type == 'ARP':
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_ARP,
                    arp_spa=ip_src,
                    arp_tpa=ip_dst
                )
                self.flows.install_once(dp, node, cookie, FLOW_PRIORITY, match, actions, idle_timeout=0)

        return ports_map[src][1]

    # -------------------- PACKET IN --------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match['in_port']
        dpid = dp.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # LLDP
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            self.topo.lldp_ports[dpid].add(in_port)
            return

        # IPv6 drop
        if eth.ethertype == 0x86dd:
            return

        src_mac = eth.src
        dst_mac = eth.dst

        trunk_ports = self.topo.trunk_ports(dpid)
        flood_ports = self.topo.flood_ports_on_tree(dpid, in_port)

        # aprender host solo si es edge port
        if in_port not in trunk_ports:
            old = self.hosts.get(src_mac)
            if old is None or old != (dpid, in_port):
                self.hosts[src_mac] = (dpid, in_port)

        out_port = None

        now = time.time()
        if now - self._last_pkt_log > self.PKT_LOG_INTERVAL:
            self._last_pkt_log = now
            self.logger.info(f"[PKTIN] sw={dpid} in={in_port} eth={hex(eth.ethertype)} src={src_mac} dst={dst_mac}")


        # -------- IP --------
        if eth.ethertype == ether_types.ETH_TYPE_IP:
            nw = pkt.get_protocol(ipv4.ipv4)
            if nw is None:
                return
            
            self.logger.info(f"[IP] {nw.src} -> {nw.dst} sw={dpid} in={in_port}")

            src_ip, dst_ip = nw.src, nw.dst

            self.arp_table[src_ip] = src_mac

            if src_mac in self.hosts and dst_mac in self.hosts:
                h1 = self.hosts[src_mac]
                h2 = self.hosts[dst_mac]
                out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip, 'IP')
                # reverse
                self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip, 'IP')

        # -------- ARP --------
        elif eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt is None:
                return

            self.logger.info(f"[ARP] op={arp_pkt.opcode} {arp_pkt.src_ip}->{arp_pkt.dst_ip} sw={dpid} in={in_port}")

            src_ip = arp_pkt.src_ip
            dst_ip = arp_pkt.dst_ip

            self.arp_table[src_ip] = src_mac

            if arp_pkt.opcode == arp.ARP_REPLY:
                if src_mac in self.hosts and dst_mac in self.hosts:
                    h1 = self.hosts[src_mac]
                    h2 = self.hosts[dst_mac]
                    out_port = self.install_paths(h1[0], h1[1], h2[0], h2[1], src_ip, dst_ip, 'ARP')
                    self.install_paths(h2[0], h2[1], h1[0], h1[1], dst_ip, src_ip, 'ARP')

            # ARP_REQUEST: por ahora flood controlado por spanning tree (baseline)
            elif arp_pkt.opcode == arp.ARP_REQUEST:
                pass

        # si no hay out_port calculado -> flooding controlado
        if out_port is None:
            if flood_ports:
                actions = [parser.OFPActionOutput(p) for p in flood_ports]
            else:
                return
        else:
            actions = [parser.OFPActionOutput(out_port)]

        data = None
        if msg.buffer_id == ofp.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        dp.send_msg(out)

    # -------------------- SWITCH FEATURES --------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # Drop IPv6
        match_ipv6 = parser.OFPMatch(eth_type=0x86dd)
        # cookie 0 (reglas base)
        self.flows.add_flow(dp, FLOW_PRIORITY, match_ipv6, [], idle_timeout=0, cookie=0)

        # default -> controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.flows.add_flow(dp, 0, match, actions, idle_timeout=0, cookie=0)

    # -------------------- TOPOLOGY EVENTS --------------------
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        dp = ev.switch.dp
        self.topo.on_switch_enter(dp)

        # si el switch reinicia, tu cache puede quedar desync => safe reset de cache local
        self.flows.forget_all()

    @set_ev_cls(event.EventSwitchLeave, MAIN_DISPATCHER)
    def switch_leave_handler(self, ev):
        dpid = ev.switch.dp.id
        self.topo.on_switch_leave(dpid)

    @set_ev_cls(event.EventLinkAdd, MAIN_DISPATCHER)
    def link_add_handler(self, ev):
        self.topo.on_link_add(
            ev.link.src.dpid, ev.link.src.port_no,
            ev.link.dst.dpid, ev.link.dst.port_no
        )

    @set_ev_cls(event.EventLinkDelete, MAIN_DISPATCHER)
    def link_delete_handler(self, ev):
        s1, s2 = ev.link.src.dpid, ev.link.dst.dpid

        # update topology
        self.topo.on_link_delete(s1, s2)

        # delete flows affected by that link
        self.flows.delete_cookies_using_link(s1, s2)

        # invalidate cached paths using that link (on-demand recompute)
        self.paths.invalidate_by_link(s1, s2)
