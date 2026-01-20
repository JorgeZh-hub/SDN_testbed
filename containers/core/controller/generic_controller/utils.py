# utils.py
from __future__ import annotations
import time
import zlib
import ipaddress
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Iterable

APP_COOKIE_MASK_FULL = 0xFFFFFFFFFFFFFFFF

def now() -> float:
    return time.time()

def stable_u32_hash(obj: Any) -> int:
    # Deterministic 32-bit hash (avoid Python's randomized hash())
    data = repr(obj).encode("utf-8", "ignore")
    return zlib.crc32(data) & 0xFFFFFFFF

def dscp_from_ipv4_tos(tos: int) -> int:
    # DSCP is the upper 6 bits of IPv4 TOS (DSCP|ECN)
    return (tos & 0xFC) >> 2

def ip_in_any_subnet(ip_str: str, subnets: Iterable[str]) -> bool:
    ip_obj = ipaddress.ip_address(ip_str)
    for s in subnets:
        if ip_obj in ipaddress.ip_network(s, strict=False):
            return True
    return False

def mac_oui(mac_str: str) -> str:
    # Returns "AA:BB:CC"
    parts = mac_str.split(":")
    if len(parts) < 3:
        return ""
    return ":".join([p.upper().zfill(2) for p in parts[:3]])

def normalize_sw_port_name(name: str) -> str:
    # Accepts "switch1-eth1" or "s1-eth1"; returns a canonical-like "s1-eth1" when possible.
    n = name.strip()
    n_low = n.lower()
    # convert "switchX-ethY" -> "sX-ethY"
    if n_low.startswith("switch"):
        rest = n_low[len("switch"):]
        return "s" + rest
    return n_low

@dataclass(frozen=True)
class FlowKey:
    eth_type: int
    src_mac: str
    dst_mac: str
    ip_src: Optional[str]
    ip_dst: Optional[str]
    ip_proto: Optional[int]
    l4_src: Optional[int]
    l4_dst: Optional[int]

@dataclass
class FlowDescriptor:
    # Endpoints in the switch graph
    src_dpid: int
    dst_dpid: int

    # L2/L3/L4 identification (directional)
    key: FlowKey

    # Optional DSCP (IPv4)
    dscp: Optional[int] = None
