#!/usr/bin/env python3
"""
Parse capture_conf.yml and topology_conf.yml to derive:
  - switch interfaces to capture (from flows link endpoints)
  - containers to capture metrics
  - links to monitor

Outputs lines to stdout:
  IFACE=<switch-iface>
  CONTAINER=<container-name>
  LINK=<link-desc>
"""
import argparse
from pathlib import Path
import yaml


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def find_switch_iface_for_host(links, host_name: str):
    for item in links:
        if not isinstance(item, dict):
            continue
        for _, desc in item.items():
            parts = [p.strip() for p in str(desc).split(",")]
            if len(parts) < 2:
                continue
            host_if, sw_if = parts[0], parts[1]
            if host_if.startswith(host_name + "-"):
                return sw_if
    raise ValueError(f"No se encontró interfaz de switch para host {host_name}")


def derive_interfaces(capture_conf, topo_conf):
    interfaces = set()
    links = topo_conf.get("links", [])

    for flow in capture_conf.get("flows", []):
        link = flow.get("link", "")
        if "-" not in link:
            raise ValueError(f"Formato de link inválido en flujo {flow.get('name')}: {link}")
        host_src, host_dst = link.split("-", 1)
        for host in (host_src, host_dst):
            iface = find_switch_iface_for_host(links, host)
            interfaces.add(iface)
    return interfaces


def derive_links(capture_conf):
    normalized = []
    for l in capture_conf.get("links", []):
        if isinstance(l, dict):
            for _, val in l.items():
                normalized.append(str(val).strip())
        else:
            normalized.append(str(l).strip())
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Deriva interfaces, contenedores y enlaces desde capture_conf.yml")
    parser.add_argument("--capture-conf", required=True, help="Ruta a capture_conf.yml")
    parser.add_argument("--topology", required=True, help="Ruta a topology_conf.yml")
    args = parser.parse_args()

    capture_conf = load_yaml(Path(args.capture_conf))
    topo_conf = load_yaml(Path(args.topology))

    interfaces = derive_interfaces(capture_conf, topo_conf)
    containers = set(str(c) for c in capture_conf.get("containers", []) if c)
    links = derive_links(capture_conf)

    for iface in sorted(interfaces):
        print(f"IFACE={iface}")
    for cont in sorted(containers):
        print(f"CONTAINER={cont}")
    for link in links:
        print(f"LINK={link}")


if __name__ == "__main__":
    main()
