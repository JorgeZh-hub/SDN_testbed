# classifier.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import ip_in_any_subnet, mac_oui

@dataclass
class Rule:
    name: str
    # Logical class (free name): e.g. "HI-REL", "LOW-LAT", "BEST-EFF".
    # For compatibility, you can keep using "CRIT"/"BE".
    out_class: str

    # Match criteria for the rule (can be empty for the default).
    match: Dict[str, Any] = field(default_factory=dict)

    # Priority to order rules (higher wins).
    priority: int = 0

    # QoS: queue id suggested by the rule. If None, it can be resolved by an external
    # class->queue map or defaults.
    queue_id: Optional[int] = None

    # If True, when installing the flow it tries to use a stable L4 port (dst/src)
    # defined by the rule to avoid matching ephemeral ports. By default it keeps
    # the old behavior: CRIT => True, the rest => False.
    stable_l4: bool = False

    # If True, this class/rule is eligible for TE (reroute).
    # If None, the decision can come from a global list (managed_classes).
    te: Optional[bool] = None


@dataclass
class Policy:
    out_class: str
    queue_id: Optional[int]
    stable_side: Optional[str]
    stable_port: Optional[int]
    rule_name: Optional[str]
    te: Optional[bool]

class PriorityClassifier:
    """Rule-based classification (first-match), loaded from YAML.

    Supports optional fields in rule.match:
      - src_subnets: ["10.0.0.0/24", ...]
      - dst_subnets: [...]
      - dscp: [46, 40]  (IPv4 DSCP)
      - src_oui: ["AA:BB:CC", ...]
      - dst_oui: [...]
      - any_oui: [...]
      - ip_proto: "udp"|"tcp"|"icmp" or number (17,6,...)
      - l4_src_ports: [5683, ...]
      - l4_dst_ports: [...]
      - l4_any_ports: [...]

    Note: if no rule matches => BE.
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
                out_class = str(r.get("class", "BE")).strip().upper()
                match = r.get("match", {}) if isinstance(r.get("match", {}), dict) else {}
                prio = int(r.get("priority", 0))
                queue_id = r.get("queue", None)
                try:
                    queue_id = int(queue_id) if queue_id is not None else None
                except Exception:
                    queue_id = None

                stable_l4 = r.get("stable_l4", None)
                if stable_l4 is None:
                    # backward-compatible default
                    stable_l4 = (out_class == "CRIT")
                stable_l4 = bool(stable_l4)

                te_flag = r.get("te", None)
                te_flag = None if te_flag is None else bool(te_flag)

                rules.append(Rule(
                    name=name,
                    out_class=out_class,
                    queue_id=queue_id,
                    stable_l4=stable_l4,
                    te=te_flag,
                    match=match,
                    priority=prio,
                ))
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
        p = self.classify_policy(src_mac, dst_mac, ip_src, ip_dst, ip_proto, l4_src, l4_dst, dscp)
        return p.out_class

    def classify_policy(self,
                        src_mac: str,
                        dst_mac: str,
                        ip_src: Optional[str],
                        ip_dst: Optional[str],
                        ip_proto: Optional[int],
                        l4_src: Optional[int],
                        l4_dst: Optional[int],
                        dscp: Optional[int]) -> Policy:
        """Classification + QoS/TE metadata.

        Returns Policy with:
          - out_class: string
          - queue_id: (optional) queue suggested by the rule
          - stable_side/port: (optional) stable L4 port if applicable
          - rule_name: matched rule name
          - te: optional TE eligibility flag
        """
        for rule in self.rules:
            if self._match_rule(rule.match, src_mac, dst_mac, ip_src, ip_dst, ip_proto, l4_src, l4_dst, dscp):
                stable_side = None
                stable_port = None

                # Stable port only if stable_l4 is enabled and we are in TCP/UDP
                if rule.stable_l4 and ip_proto in (6, 17):
                    m = rule.match or {}

                    # If the rule explicitly sets dst or src, that wins.
                    if "l4_dst_ports" in m and l4_dst is not None:
                        stable_side, stable_port = "dst", int(l4_dst)
                    elif "l4_src_ports" in m and l4_src is not None:
                        stable_side, stable_port = "src", int(l4_src)
                    elif "l4_any_ports" in m:
                        allowed = [int(x) for x in m.get("l4_any_ports", [])]

                        # Prefer dst if both match
                        if l4_dst is not None and int(l4_dst) in allowed:
                            stable_side, stable_port = "dst", int(l4_dst)
                        elif l4_src is not None and int(l4_src) in allowed:
                            stable_side, stable_port = "src", int(l4_src)

                return Policy(
                    out_class=rule.out_class,
                    queue_id=rule.queue_id,
                    stable_side=stable_side,
                    stable_port=stable_port,
                    rule_name=rule.name,
                    te=rule.te,
                )

        return Policy(out_class="BE", queue_id=None, stable_side=None, stable_port=None, rule_name=None, te=None)

    def classify_detail(self,
                        src_mac: str,
                        dst_mac: str,
                        ip_src: Optional[str],
                        ip_dst: Optional[str],
                        ip_proto: Optional[int],
                        l4_src: Optional[int],
                        l4_dst: Optional[int],
                        dscp: Optional[int]) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
        """
        Returns:
          (out_class, stable_side, stable_port, rule_name)
        stable_side: "src" | "dst" | None
        stable_port: int | None
        """
        p = self.classify_policy(src_mac, dst_mac, ip_src, ip_dst, ip_proto, l4_src, l4_dst, dscp)
        return p.out_class, p.stable_side, p.stable_port, p.rule_name

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
