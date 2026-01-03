#!/usr/bin/python3
from __future__ import annotations

import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from ryu.lib.packet import tcp, udp, icmp

from ryu.topology import event

from ryu import cfg

from .utils import FlowDescriptor
from .flow_manager import FlowManager
from .topology_manager import TopologyManager
from .path_engine import PathEngine
from .stats_manager import StatsManager
from .te_engine import TEEngine
from .config_loader import load_yaml, parse_links_capacity
from .classifier import PriorityClassifier

# ---------------- CLI options ----------------
CONF = cfg.CONF
CONF.register_opts([
    cfg.StrOpt("top_config", default="", help="YAML with links"),
    cfg.StrOpt("te_config", default="", help="Classifiers"),
    cfg.FloatOpt("te_period", default=5.0),
    cfg.FloatOpt("portstats_period", default=1.0),
    cfg.FloatOpt("flowstats_period", default=5.0),
    cfg.FloatOpt("diagnostic_period", default=60.0),
    cfg.BoolOpt("observe_net", default=True, help="Request Port/FlowStats"),
    cfg.BoolOpt("te_activate", default=True, help="Run TE (te.run_once)"),
    cfg.BoolOpt("logger_stats", default=True, help="Enable StatsManager logs"),
    cfg.BoolOpt("logger_flow_manager", default=True, help="Enable FlowManager logs"),
    cfg.BoolOpt("logger_te", default=True, help="Enable TEEngine logs"),
    cfg.BoolOpt("logger_topology_manager", default=True, help="Enable topology-related logs"),
    cfg.FloatOpt("default_link_capacity", default=100.0),
    cfg.FloatOpt("hot_th", default=0.85),
    cfg.IntOpt("hot_n", default=2),
    cfg.FloatOpt("alpha_link", default=0.3),
    cfg.FloatOpt("alpha_flow", default=0.3),
    cfg.FloatOpt("cooldown", default=45.0),
    cfg.FloatOpt("K", default=5.0),
    cfg.FloatOpt("safety_factor", default=1.2),
    cfg.FloatOpt("delta_hot", default=0.05),
    cfg.FloatOpt("delta_global", default=0.02),
    cfg.FloatOpt("r_min_mbps", default=0.1),
    cfg.BoolOpt("queues_enable", default=False),
    cfg.IntOpt("app_id", default=0xBEEF),
    cfg.IntOpt("flow_priority", default=20000),
    cfg.BoolOpt("barrier", default=True),
])


class ReactiveIoTTE13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- config ---
        top_path = CONF.top_config
        te_path = CONF.te_config

        classifier_enable = bool(CONF.queues_enable or CONF.te_activate)
        te_data = load_yaml(te_path) if (te_path and classifier_enable) else {}
        top_data = load_yaml(top_path) if top_path else {}
        csv_dir = os.path.dirname(top_path) if top_path else None

        self.log_topo = CONF.logger_topology_manager

        self.cap_db = parse_links_capacity(top_data)
        self.classifier = PriorityClassifier.from_cfg(te_data) if classifier_enable else PriorityClassifier([])

        # ---- QoS mapping (genérico) ----
        qos_cfg = te_data.get("qos", {}) if isinstance(te_data, dict) else {}
        if not isinstance(qos_cfg, dict):
            qos_cfg = {}

        class_to_queue = qos_cfg.get("class_queues", qos_cfg.get("class_to_queue", {}))
        if not isinstance(class_to_queue, dict):
            class_to_queue = {}
        default_queue_id = qos_cfg.get("default_queue", None)
        try:
            default_queue_id = int(default_queue_id) if default_queue_id is not None else None
        except Exception:
            default_queue_id = None

        # ---- TE: clases elegibles para reroute ----
        te_cfg = te_data.get("te", {}) if isinstance(te_data, dict) else {}
        if not isinstance(te_cfg, dict):
            te_cfg = {}
        managed_classes = te_cfg.get("managed_classes", None)
        if not (isinstance(managed_classes, list) and managed_classes):
            # fallback: si hay reglas con te: true, usar esas clases
            mc = sorted({r.out_class for r in getattr(self.classifier, "rules", []) if getattr(r, "te", None) is True})
            managed_classes = mc if mc else None


        # ---- TE: prioridad por colas ----
        protect_queues = te_cfg.get("protect_queues", None)
        if protect_queues is not None and not isinstance(protect_queues, list):
            protect_queues = None

        lower_queue_is_higher_priority = te_cfg.get("lower_queue_is_higher_priority", True)
        lower_queue_is_higher_priority = bool(lower_queue_is_higher_priority)

        unknown_queue_behavior = te_cfg.get("unknown_queue_behavior", "protect")
        self.topo = TopologyManager(self.logger)
        self.flow_mgr = FlowManager(
            topo=self.topo,
            logger=self.logger,
            app_id=CONF.app_id,
            priority_base=CONF.flow_priority,
            barrier=CONF.barrier,
            queues_enable=CONF.queues_enable,
            class_to_queue=class_to_queue,
            default_queue_id=default_queue_id,
            classifier=self.classifier,
            log_enabled=CONF.logger_flow_manager,
        )
        self.stats = StatsManager(
            topo=self.topo,
            flow_mgr=self.flow_mgr,
            logger=self.logger,
            default_capacity_mbps=CONF.default_link_capacity,
            alpha_link=CONF.alpha_link,
            alpha_flow=CONF.alpha_flow,
            hot_th=CONF.hot_th,
            hot_n=CONF.hot_n,
            portstats_period=CONF.portstats_period,
            flowstats_period=CONF.flowstats_period,
            app_id=CONF.app_id,
            log_enabled=CONF.logger_stats,
            csv_dir=csv_dir,
        )
        self.path_engine = PathEngine(self.topo, self.stats, self.logger)
        self.te = TEEngine(
            topo=self.topo,
            stats=self.stats,
            path_engine=self.path_engine,
            flow_mgr=self.flow_mgr,
            logger=self.logger,
            te_period=CONF.te_period,
            cooldown_s=CONF.cooldown,
            hot_th=CONF.hot_th,
            delta_hot=CONF.delta_hot,
            delta_global=CONF.delta_global,
            K=CONF.K,
            safety_factor=CONF.safety_factor,
            r_min_mbps=CONF.r_min_mbps,
            table_id=0,
            managed_classes=managed_classes,
            protect_queues=protect_queues,
            lower_queue_is_higher_priority=lower_queue_is_higher_priority,
            unknown_queue_behavior=unknown_queue_behavior,
            log_enabled=CONF.logger_te,
        )

        self.logger.info(
            "[INIT] ReactiveIoTTE13 top_config=%s te_config=%s rules=%d caps=%d",
            top_path or "-",
            te_path or "-",
            len(self.classifier.rules),
            len(self.cap_db.cap_by_portpair),
        )
        if managed_classes is not None:
            self.logger.info("[INIT] TE managed_classes=%s", sorted({str(x).strip().upper() for x in managed_classes}))
        self.logger.info("[INIT] TE protect_queues=%s lower_queue_is_higher_priority=%s unknown_queue_behavior=%s",
                         sorted({int(x) for x in (protect_queues or [])}) if protect_queues is not None else [0],
                         lower_queue_is_higher_priority,
                         unknown_queue_behavior)
        if classifier_enable:
            self.logger.info(
                "[INIT] QoS policy_enabled=%s set_queue_actions=%s default_queue=%s class_queues=%s",
                classifier_enable,
                CONF.queues_enable,
                default_queue_id if default_queue_id is not None else "-",
                {str(k).strip().upper(): int(v) for k, v in class_to_queue.items()} if class_to_queue else "-",
            )
        self.logger.info(
            "[INIT] periods te=%ss portstats=%ss flowstats=%ss app_id=0x%X prio=%d",
            CONF.te_period,
            CONF.portstats_period,
            CONF.flowstats_period,
            CONF.app_id,
            CONF.flow_priority,
        )
        self.logger.info(
            "[INIT] flags observe_net=%s te_activate=%s",
            CONF.observe_net,
            CONF.te_activate,
        )
        if CONF.te_activate and not CONF.observe_net:
            self.logger.warning("[INIT] te_activate enabled but observe_net disabled: TE will lack network metrics")

        self._last_pkt_log = 0.0
        self.PKT_LOG_INTERVAL = 2.0

        self._monitor_thread = hub.spawn(self._monitor_loop)

    # ---------------- Monitor loop ----------------
    def _monitor_loop(self):
        self.logger.info("[MONITOR] loop started")
        # Two periodic loops in one thread to keep things simple
        last_port = 0.0
        last_flow = 0.0
        last_te = 0.0
        last_diag = 0.0
        while True:
            t = time.time()
            if CONF.observe_net and (t - last_port >= CONF.portstats_period):
                last_port = t
                self.stats.request_port_stats()
            if CONF.observe_net and (t - last_flow >= CONF.flowstats_period):
                last_flow = t
                self.stats.request_flow_stats()
            if CONF.te_activate and (t - last_te >= CONF.te_period):
                last_te = t
                try:
                    self.te.run_once()
                except Exception as e:
                    self.logger.exception("TE error: %s", e)
            if t - last_diag >= CONF.diagnostic_period:
                last_diag = t
                dpids = sorted(self.topo.datapaths.keys())
                tree_ports = {dpid: sorted(ports) for dpid, ports in self.topo.tree_ports.items()}
                if self.log_topo:
                    self.logger.info(
                        "[NET] dps=%s links=%s tree_root=%s tree_ports=%s",
                        dpids,
                        len(self.topo.link_ports),
                        self.topo.tree_root,
                        tree_ports,
                    )
                edge_stats = []
                for e, (u_p, v_p) in self.topo.link_ports.items():
                    u, v = e
                    cap = self.stats.capacity_mbps(e)
                    load = self.stats.link_load_mbps.get(e, 0.0)
                    util = (load / cap) if cap > 0 else 0.0
                    u_name = self.topo.get_port_name(u, u_p) if u_p is not None else ""
                    v_name = self.topo.get_port_name(v, v_p) if v_p is not None else ""
                    hot = self.stats.is_hot(e)
                    edge_stats.append((util, f"{u}->{v}", u_name, v_name, cap, load, hot))
                edge_stats.sort(key=lambda x: x[0], reverse=True)
                top_edges = edge_stats[:10]
                if self.log_topo:
                    self.logger.info(
                        "[LINKS] top_util=%s flows=%s",
                        top_edges,
                        len(self.flow_mgr.cookie_path),
                    )
                hot_edges = [e for e in edge_stats if e[6]]
                if hot_edges:
                    if self.log_topo:
                        self.logger.info("[LINKS] hot=%s", hot_edges)
            hub.sleep(0.2)

    # ---------------- Switch lifecycle ----------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.topo.register_dp(dp)
        if self.log_topo:
            self.logger.info("[SWITCH] features dpid=%s", dp.id)

        # table-miss (send to controller)
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, table_id=0 ,priority=0, match=match, instructions=inst)
        dp.send_msg(mod)
        if self.log_topo:
            self.logger.info("[SWITCH] installed table-miss dpid=%s", dp.id)

        # request PortDesc to learn port names
        req = parser.OFPPortDescStatsRequest(dp, 0)
        dp.send_msg(req)
        if self.log_topo:
            self.logger.info("[SWITCH] sent PortDesc request dpid=%s", dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.topo.register_dp(dp)
        elif ev.state == DEAD_DISPATCHER:
            self.topo.unregister_dp(dp.id)
            if self.log_topo:
                self.logger.warning("[SWITCH] disconnected dpid=%s", dp.id)

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_reply_handler(self, ev):
        dp = ev.msg.datapath
        self.topo.update_port_desc(dp.id, ev.msg.body)
        # Re-intenta mapear capacidades para links ya descubiertos (por si LinkAdd llegó antes de PortDesc)
        for (u,v), (u_p, v_p) in list(self.topo.link_ports.items()):
            spn = self.topo.get_port_name(u, u_p) or ""
            dpn = self.topo.get_port_name(v, v_p) or ""
            cap = self.cap_db.capacity_for(spn, dpn)
            if cap is not None:
                self.stats.set_capacity((u,v), cap)
                self.stats.set_capacity((v,u), cap)

    # ---------------- Topology events (LLDP discovery) ----------------
    @set_ev_cls(event.EventLinkAdd)
    def _link_add(self, ev):
        s = ev.link.src
        d = ev.link.dst
        self.topo.add_link(s.dpid, s.port_no, d.dpid, d.port_no)
        if self.log_topo:
            self.logger.info(
                "[LINK] add %s:%s -> %s:%s",
                s.dpid,
                s.port_no,
                d.dpid,
                d.port_no,
            )

        # Try to map capacity from YAML via port names
        spn = self.topo.get_port_name(s.dpid, s.port_no) or ""
        dpn = self.topo.get_port_name(d.dpid, d.port_no) or ""
        cap = self.cap_db.capacity_for(spn, dpn)
        if cap is not None:
            self.stats.set_capacity((s.dpid, d.dpid), cap)
            self.stats.set_capacity((d.dpid, s.dpid), cap)

    @set_ev_cls(event.EventLinkDelete)
    def _link_del(self, ev):
        s = ev.link.src
        d = ev.link.dst
        self.topo.del_link(s.dpid, d.dpid)
        if self.log_topo:
            self.logger.warning(
                "[LINK] del %s:%s -> %s:%s",
                s.dpid,
                s.port_no,
                d.dpid,
                d.port_no,
            )

        # Optional: clean cookies affected by this link (simple)
        e1 = (s.dpid, d.dpid)
        e2 = (d.dpid, s.dpid)
        affected = set(self.flow_mgr.link_cookies.get(e1, set())) | set(self.flow_mgr.link_cookies.get(e2, set()))
        for c in affected:
            path = self.flow_mgr.cookie_path.get(c, None)
            if path:
                self.flow_mgr.delete_cookie_exact(c, dpids=path)
            self.flow_mgr._unregister_cookie(c)

    # ---------------- Stats replies ----------------
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply(self, ev):
        self.stats.handle_port_stats_reply(ev.msg.datapath, ev.msg)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        self.stats.handle_flow_stats_reply(ev.msg.datapath, ev.msg)

    # ---------------- PacketIn baseline ----------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        in_port = msg.match.get("in_port", 0)
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        # IPv6 drop
        if eth.ethertype == 0x86dd:
            return
        if eth is None:
            return
        if eth.ethertype == 0x88cc:
            return

        dpid = dp.id
        
        now = time.time()
        if now - self._last_pkt_log >= self.PKT_LOG_INTERVAL:
            self._last_pkt_log = now
            self.logger.info(
                "[PKTIN] sw=%s in=%s eth=%s src=%s dst=%s",
                dpid,
                in_port,
                hex(eth.ethertype),
                eth.src,
                eth.dst,
            )
        self.topo.learn_mac(dpid, eth.src, in_port)
        self.flow_mgr.learn_host(dpid, eth.src, in_port)

        # ARP: flood via tree
        a = pkt.get_protocol(arp.arp)
        if a:
            if a.opcode == arp.ARP_REQUEST:
                out_ports = self.topo.flood_ports_on_tree(dpid, in_port)
                actions = [parser.OFPActionOutput(p) for p in out_ports]
                out = parser.OFPPacketOut(datapath=dp, buffer_id=ofp.OFP_NO_BUFFER,
                                        in_port=in_port, actions=actions, data=msg.data)
                
                #self.logger.info("Action taked: %s", actions)
                dp.send_msg(out)
                return

        ip4 = pkt.get_protocol(ipv4.ipv4)
        if ip4 is None:
            # Non-IP: simple L2 forwarding (ARP-REPLIES)
            out_port = self.topo.lookup_mac_port(dpid, eth.dst)
            if out_port is None:
                out_ports = self.topo.flood_ports_on_tree(dpid, in_port)
                actions = [parser.OFPActionOutput(p) for p in out_ports]
            else:
                actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(datapath=dp, buffer_id=ofp.OFP_NO_BUFFER,
                                      in_port=in_port, actions=actions, data=msg.data)
            dp.send_msg(out)
            return

        # IP flow: parse L4
        l4 = pkt.get_protocol(tcp.tcp) or pkt.get_protocol(udp.udp) or pkt.get_protocol(icmp.icmp)

        # resolve src/dst switch (L2 learning)
        src_sw = dpid
        dst_sw = None
        hosts_snapshot = dict(self.flow_mgr.hosts)
        #self.logger.info("[L2] lookup dst_mac=%s hosts=%s", eth.dst, hosts_snapshot)
        # find dst MAC location across switches (best effort) using hosts first
        if eth.dst in self.flow_mgr.hosts:
            dst_sw = self.flow_mgr.hosts[eth.dst][0]
        else:
            for sw, table in self.topo.mac_to_port.items():
                if eth.dst in table:
                    dst_sw = sw
                    break
        if dst_sw is None:
            # unknown -> flood
            out_ports = self.topo.flood_ports_on_tree(dpid, in_port)
            self.logger.info(
                "[L2] dst_mac unknown; mac_table=%s hosts=%s flood_ports=%s",
                dict(self.topo.mac_to_port),
                dict(self.flow_mgr.hosts),
                out_ports,
            )
            actions = [parser.OFPActionOutput(p) for p in out_ports]
            out = parser.OFPPacketOut(datapath=dp, buffer_id=ofp.OFP_NO_BUFFER,
                                      in_port=in_port, actions=actions, data=msg.data)
            dp.send_msg(out)
            return

        fk = self.flow_mgr.build_flow_key(eth, ip4, l4)
        dscp = (ip4.tos & 0xFC) >> 2
        desc = FlowDescriptor(src_dpid=src_sw, dst_dpid=dst_sw, key=fk, dscp=dscp)
        cookie = self.flow_mgr.get_or_create_cookie(desc)

        # install baseline path (hop-count == cost 1)
        path = self.path_engine.shortest_path(src_sw, dst_sw)
        if not path:
            self.logger.warning("[PATH] no path src=%s dst=%s", src_sw, dst_sw)
            return
        

        already = (cookie in self.flow_mgr.cookie_path and self.flow_mgr.cookie_path[cookie] == path)
        if not already:
            self.flow_mgr.install_path(
                desc,
                cookie,
                path,
                self.flow_mgr.hosts[eth.dst][1],
                table_id=0,
            )
        else:
            self.logger.debug("[FLOW] skip reinstall cookie=%s path=%s", hex(cookie), path)

    @set_ev_cls(ofp_event.EventOFPErrorMsg, MAIN_DISPATCHER)
    def error_msg_handler(self, ev):
        msg = ev.msg
        self.logger.error("[OF-ERROR] dpid=%s type=0x%x code=0x%x data=%s",
                        msg.datapath.id, msg.type, msg.code, msg.data.hex())
