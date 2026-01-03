# classifier.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import ip_in_any_subnet, mac_oui

@dataclass
class Rule:
    name: str
    # Clase lógica (nombre libre): p.ej. "HI-REL", "LOW-LAT", "BEST-EFF".
    # Por compatibilidad, puedes seguir usando "CRIT"/"BE".
    out_class: str

    # Match asociado a la regla (puede ser vacío en el "default").
    match: Dict[str, Any] = field(default_factory=dict)

    # Prioridad para ordenar reglas (más alto gana).
    priority: int = 0

    # QoS: cola (queue id) sugerida por la regla. Si es None, se puede
    # resolver por mapeo externo (class->queue) o por defaults.
    queue_id: Optional[int] = None

    # Si True, al instalar el flow se intentará usar un puerto L4 estable
    # (dst/src) definido por la regla, para evitar matchear contra puertos
    # efímeros. Por defecto se mantiene el comportamiento antiguo:
    # CRIT => True, el resto => False.
    stable_l4: bool = False

    # Si True, esta clase/regla se considera elegible para TE (reroute).
    # Si es None, la decisión puede venir de una lista global (managed_classes).
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
        """Clasificación + metadatos QoS/TE.

        Retorna Policy con:
          - out_class: string
          - queue_id: (opcional) queue sugerida por la regla
          - stable_side/port: (opcional) puerto L4 estable si aplica
          - rule_name: nombre de la regla matcheada
          - te: flag opcional de elegibilidad TE
        """
        for rule in self.rules:
            if self._match_rule(rule.match, src_mac, dst_mac, ip_src, ip_dst, ip_proto, l4_src, l4_dst, dscp):
                stable_side = None
                stable_port = None

                # Puerto estable solo si se habilita stable_l4 y estamos en TCP/UDP
                if rule.stable_l4 and ip_proto in (6, 17):
                    m = rule.match or {}

                    # Si la regla define explícitamente dst o src, eso manda.
                    if "l4_dst_ports" in m and l4_dst is not None:
                        stable_side, stable_port = "dst", int(l4_dst)
                    elif "l4_src_ports" in m and l4_src is not None:
                        stable_side, stable_port = "src", int(l4_src)
                    elif "l4_any_ports" in m:
                        allowed = [int(x) for x in m.get("l4_any_ports", [])]

                        # Preferimos dst si ambos cumplen
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
        Retorna:
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
