#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_switch_iface_for_host(links, host_name):
    for item in links:
        if isinstance(item, dict):
            for _, desc in item.items():
                parts = [p.strip() for p in desc.split(",")]
                if len(parts) < 2:
                    continue
                host_if, sw_if = parts[0], parts[1]
                if host_if.startswith(host_name + "-"):
                    return sw_if
    raise ValueError(f"No se encontró interfaz de switch para host {host_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Deriva las interfaces de switch a capturar desde flows.yml/topology."
    )
    parser.add_argument("--flows", default="flows.yml", help="Archivo YAML de flujos.")
    parser.add_argument(
        "--topology",
        default="topology_conf.yml",
        help="Archivo YAML de topología.",
    )
    args = parser.parse_args()

    flows_cfg = load_yaml(args.flows)
    topo_cfg = load_yaml(args.topology)

    links = topo_cfg["links"]

    interfaces = set()

    for flow in flows_cfg.get("flows", []):
        link = flow["link"]
        if "-" not in link:
            raise ValueError(f"Formato de link inválido en flujo {flow['name']}: {link}")
        host_src, host_dst = link.split("-", 1)
        for host in (host_src, host_dst):
            iface = find_switch_iface_for_host(links, host)
            interfaces.add(iface)

    for iface in sorted(interfaces):
        print(iface)


if __name__ == "__main__":
    main()
