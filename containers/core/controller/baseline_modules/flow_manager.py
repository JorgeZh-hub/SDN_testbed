# flow_manager.py
import time, zlib
from collections import defaultdict
from ryu.lib import hub

class FlowManager:
    def __init__(self, logger, get_dp, app_id=0x13, cookie_ttl=120, gc_interval=10):
        self.log = logger
        self.get_dp = get_dp
        self.app_id = app_id
        self.cookie_ttl = cookie_ttl
        self.gc_interval = gc_interval

        self.cookie_installed = defaultdict(set)  # cookie -> set(dpids)
        self.cookie_path_nodes = {}               # cookie -> [dpids]
        self.cookie_last_seen = {}                # cookie -> ts

        self._gc_thread = hub.spawn(self._gc)

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0, buffer_id=None, cookie=0):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id is not None and buffer_id != ofp.OFP_NO_BUFFER:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout,
                                    cookie=cookie, cookie_mask=0,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath,
                                    priority=priority, match=match,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout,
                                    cookie=cookie, cookie_mask=0,
                                    instructions=inst)
        datapath.send_msg(mod)
        return True


    def make_cookie(self, flow_type, ip_src, ip_dst):
        base = f"{flow_type}|{ip_src}|{ip_dst}"
        h = zlib.crc32(base.encode()) & 0xFFFFFFFF
        return (self.app_id << 48) | h

    def touch(self, cookie, nodes_in_path=None):
        self.cookie_last_seen[cookie] = time.time()
        if nodes_in_path is not None:
            self.cookie_path_nodes[cookie] = list(nodes_in_path)

    def install_once(self, dp, dpid, cookie, priority, match, actions, idle_timeout=0, hard_timeout=0, buffer_id=None):
        if dpid in self.cookie_installed[cookie]:
            self.log.debug(f"[FLOW SKIP] dpid={dpid} cookie={hex(cookie)}")
            return False

        ofp = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id is not None and buffer_id != ofp.OFP_NO_BUFFER:
            mod = parser.OFPFlowMod(datapath=dp, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout,
                                    cookie=cookie, cookie_mask=0,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=dp,
                                    priority=priority, match=match,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout,
                                    cookie=cookie, cookie_mask=0,
                                    instructions=inst)

        dp.send_msg(mod)
        self.log.info(
            f"[FLOW ADD] dpid={dpid} cookie={hex(cookie)} prio={priority} "
            f"match={match} actions={actions}"
        )
        self.cookie_installed[cookie].add(dpid)
        return True

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
            dp = self.get_dp(dpid)
            if dp:
                self.delete_flows_by_cookie(dp, cookie)
        self.cookie_installed.pop(cookie, None)
        self.cookie_path_nodes.pop(cookie, None)
        self.cookie_last_seen.pop(cookie, None)
        self.log.warning(f"[FLOW DEL] cookie={hex(cookie)} dpids={sorted(self.cookie_installed.get(cookie, []))}")


    @staticmethod
    def _path_uses_link(nodes, s1, s2):
        return any(
            (nodes[i] == s1 and nodes[i+1] == s2) or (nodes[i] == s2 and nodes[i+1] == s1)
            for i in range(len(nodes) - 1)
        )

    def delete_cookies_using_link(self, s1, s2):
        affected = [c for c, nodes in list(self.cookie_path_nodes.items())
                    if nodes and self._path_uses_link(nodes, s1, s2)]
        self.log.warning(f"[LINK IMPACT] link={s1}<->{s2} affected={len(affected)} cookies")
        for c in affected:
            self.delete_cookie_everywhere(c)
            self.log.warning(f"  - cookie={hex(c)} path={self.cookie_path_nodes.get(c)}")
        return affected

    def forget_all(self):
        self.cookie_installed.clear()
        self.cookie_path_nodes.clear()
        self.cookie_last_seen.clear()

    def _gc(self):
        while True:
            now = time.time()
            for c, last in list(self.cookie_last_seen.items()):
                if (now - last) > self.cookie_ttl:
                    self.delete_cookie_everywhere(c)
                    self.log.info(f"[GC] expired cookie={hex(c)}")
            hub.sleep(self.gc_interval)

