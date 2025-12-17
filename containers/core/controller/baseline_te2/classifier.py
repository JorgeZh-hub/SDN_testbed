# classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import ip_in_any_subnet, mac_oui

@dataclass
class Rule:
    name: str
    out_class: str  # "CRIT" or "BE"
    match: Dict[str, Any]
    priority: int = 0

class PriorityClassifier:
    """Clasificación por reglas (first-match), cargadas desde YAML.

    Soporta campos opcionales en rule.match:
      - src_subnets: ["10.0.0.0/24", ...]
      - dst_subnets: [...]
      - dscp: [46, 40]  (IPv4 DSCP)
      - src_oui: ["AA:BB:CC", ...]
      - dst_oui: [...]
      - any_oui: [...]
      - ip_proto: "udp"|"tcp"|"icmp" o número (17,6,...)
      - l4_src_ports: [5683, ...]
      - l4_dst_ports: [...]
      - l4_any_ports: [...]

    Nota: si ninguna regla matchea => BE.
    """

    def __init__(self, rules: List[Rule]):
        # higher priority first
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "PriorityClassifier":
        rules_cfg = cfg.get("classifiers", [])
        rules: List[Rule] = []
        if isinstance(rules_cfg, list):
            for r in rules_cfg:
                if not isinstance(r, dict):
                    continue
                name = str(r.get("name", "rule"))
                out_class = str(r.get("class", "BE")).upper()
                match = r.get("match", {}) if isinstance(r.get("match", {}), dict) else {}
                prio = int(r.get("priority", 0))
                rules.append(Rule(name=name, out_class=out_class, match=match, priority=prio))
        return PriorityClassifier(rules)

    def classify(self,
                 src_mac: str,
                 dst_mac: str,
                 ip_src: Optional[str],
                 ip_dst: Optional[str],
                 ip_proto: Optional[int],
                 l4_src: Optional[int],
                 l4_dst: Optional[int],
                 dscp: Optional[int]) -> str:
        for rule in self.rules:
            if self._match_rule(rule.match, src_mac, dst_mac, ip_src, ip_dst, ip_proto, l4_src, l4_dst, dscp):
                return rule.out_class
        return "BE"

    def _match_rule(self, m: Dict[str, Any],
                    src_mac: str, dst_mac: str,
                    ip_src: Optional[str], ip_dst: Optional[str],
                    ip_proto: Optional[int],
                    l4_src: Optional[int], l4_dst: Optional[int],
                    dscp: Optional[int]) -> bool:
        # Subnets
        if "src_subnets" in m:
            if not (ip_src and ip_in_any_subnet(ip_src, m.get("src_subnets", []))):
                return False
        if "dst_subnets" in m:
            if not (ip_dst and ip_in_any_subnet(ip_dst, m.get("dst_subnets", []))):
                return False

        # DSCP
        if "dscp" in m:
            if dscp is None or int(dscp) not in [int(x) for x in m.get("dscp", [])]:
                return False

        # MAC OUI
        src_oui = mac_oui(src_mac)
        dst_oui = mac_oui(dst_mac)
        if "src_oui" in m:
            if src_oui not in [x.upper() for x in m.get("src_oui", [])]:
                return False
        if "dst_oui" in m:
            if dst_oui not in [x.upper() for x in m.get("dst_oui", [])]:
                return False
        if "any_oui" in m:
            allowed = [x.upper() for x in m.get("any_oui", [])]
            if src_oui not in allowed and dst_oui not in allowed:
                return False

        # IP proto
        if "ip_proto" in m:
            want = m.get("ip_proto")
            if ip_proto is None:
                return False
            if isinstance(want, str):
                w = want.lower()
                table = {"tcp": 6, "udp": 17, "icmp": 1}
                want_num = table.get(w, None)
                if want_num is None or ip_proto != want_num:
                    return False
            else:
                if int(ip_proto) != int(want):
                    return False

        # Ports
        if "l4_src_ports" in m:
            if l4_src is None or int(l4_src) not in [int(x) for x in m.get("l4_src_ports", [])]:
                return False
        if "l4_dst_ports" in m:
            if l4_dst is None or int(l4_dst) not in [int(x) for x in m.get("l4_dst_ports", [])]:
                return False
        if "l4_any_ports" in m:
            allowed = [int(x) for x in m.get("l4_any_ports", [])]
            if (l4_src is None or int(l4_src) not in allowed) and (l4_dst is None or int(l4_dst) not in allowed):
                return False

        return True
