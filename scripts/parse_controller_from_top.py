#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import yaml


def normalize_volume(raw, base_dir: Path, host_root: str):
    parts = str(raw).split(":")
    host_part = parts[0]
    rest = ":".join(parts[1:]) if len(parts) > 1 else ""
    host_path = Path(host_part)
    if not host_path.is_absolute():
        host_path = (base_dir / host_path).resolve()
    if str(host_path).startswith("/sim/") and host_root:
        host_path = Path(host_root).resolve() / host_path.relative_to("/sim")
    if rest:
        return f"{host_path}:{rest}"
    return str(host_path)


def main():
    if len(sys.argv) < 2:
        return

    topo_path = Path(sys.argv[1]).resolve()
    try:
        data = yaml.safe_load(topo_path.read_text()) or {}
    except FileNotFoundError:
        return

    ctrls = data.get("controller") or data.get("controllers")
    name = None
    cfg = None
    if isinstance(ctrls, dict) and ctrls:
        # two shapes:
        # 1) {"name": "...", "image": "...", ...}
        # 2) {"my_ctrl": {"image": "...", ...}}
        lowered_keys = {str(k).lower() for k in ctrls.keys()}
        if {"image", "command", "comand", "volumes", "name"} & lowered_keys:
            cfg = ctrls
            name = cfg.get("name")
        else:
            name, cfg = next(iter(ctrls.items()))
    elif isinstance(ctrls, list) and ctrls:
        cfg = ctrls[0]
        name = cfg.get("name")
    if not cfg or not isinstance(cfg, dict):
        return

    name = cfg.get("name") or name or "ryu-ctrl"
    image = cfg.get("image", "")
    command = cfg.get("command") or cfg.get("comand") or ""
    volumes = cfg.get("volumes") or []
    base_dir = topo_path.parent
    host_root = os.environ.get("HOST_PROJECT_ROOT")

    print(f"NAME={name}")
    if image:
        print(f"IMAGE={image}")
    if command:
        print(f"COMMAND={command}")
    for v in volumes:
        norm = normalize_volume(v, base_dir, host_root)
        print(f"VOLUME={norm}")


if __name__ == "__main__":
    main()
