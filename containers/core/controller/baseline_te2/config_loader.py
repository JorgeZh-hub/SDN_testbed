# config_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .utils import normalize_sw_port_name

@dataclass
class LinkCapDB:
    # capacity by unordered port-name pair (canonicalized); value in Mbps
    cap_by_portpair: Dict[Tuple[str, str], float]

    def capacity_for(self, a_port: str, b_port: str) -> Optional[float]:
        a = normalize_sw_port_name(a_port)
        b = normalize_sw_port_name(b_port)
        key = tuple(sorted((a, b)))
        return self.cap_by_portpair.get(key)

def load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML no está instalado. Instala con: pip3 install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("El YAML raíz debe ser un dict.")
    return data

def parse_links_capacity(cfg: Dict[str, Any]) -> LinkCapDB:
    # Expected:
    # links:
    #   - link1: "switch1-eth1, switch2-eth1, 30"
    cap: Dict[Tuple[str, str], float] = {}
    links = cfg.get("links", [])
    if not isinstance(links, list):
        return LinkCapDB({})

    for item in links:
        if not isinstance(item, dict):
            continue
        for _, val in item.items():
            if not isinstance(val, str):
                continue
            parts = [p.strip() for p in val.split(",")]
            if len(parts) < 3:
                continue
            a, b, c = parts[0], parts[1], parts[2]
            try:
                c_mbps = float(c)
            except Exception:
                continue
            a_n = normalize_sw_port_name(a)
            b_n = normalize_sw_port_name(b)
            key = tuple(sorted((a_n, b_n)))
            cap[key] = c_mbps

    return LinkCapDB(cap)
