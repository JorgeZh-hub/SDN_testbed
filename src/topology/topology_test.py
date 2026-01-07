#!/usr/bin/python3
# Compatible con Python 3.6.9

from mininet.net import Containernet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.clean import cleanup

from burst_mod import run_burst

from subprocess import PIPE
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import subprocess
import time
import yaml
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

setLogLevel("info")

TOPOLOGY_FILE = os.environ.get("TOPOLOGY_FILE", "topology_conf1_v2.yml")
BASE_DIR = Path(TOPOLOGY_FILE).resolve().parent

IS_ROOT = (os.geteuid() == 0)
SUDO = [] if IS_ROOT else ["sudo"]


def run_cmd(cmd):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if p.returncode != 0:
        raise RuntimeError(
            "cmd failed rc={}\ncmd={}\nstdout={}\nstderr={}".format(
                p.returncode, " ".join(cmd), p.stdout, p.stderr
            )
        )
    return p


def load_topology(path=TOPOLOGY_FILE):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_volumes(vol_list):
    resolved = []
    for v in vol_list:
        host, cont, *mode = v.split(":")
        host_path = Path(host)
        if not host_path.is_absolute():
            host_path = (BASE_DIR / host_path).resolve()

        # Remapeo /sim/... -> HOST_PROJECT_ROOT
        if str(host_path).startswith("/sim/"):
            host_root_env = os.environ.get("HOST_PROJECT_ROOT")
            if not host_root_env:
                raise ValueError("HOST_PROJECT_ROOT must be set to remap /sim paths for Docker volumes")
            host_root = Path(host_root_env).resolve()
            host_path = host_root / host_path.relative_to("/sim")

        resolved.append(":".join([str(host_path), cont] + mode))
    return resolved


def _bps_from_ratio(cap_mbps, ratio):
    if ratio is None:
        return None
    return int(float(ratio) * float(cap_mbps) * 1000000.0)


def should_apply_qos(node1_name, node2_name, switches, qos_cfg):
    """
    apply_scope:
      - "switch-switch" (default): solo enlaces switch-switch (más rápido).
      - "all-switch-ports": todo puerto donde el nodo sea switch (incluye host-switch).
    """
    scope = str(qos_cfg.get("apply_scope", "switch-switch")).strip().lower()
    n1_is_sw = (node1_name in switches)
    n2_is_sw = (node2_name in switches)

    if scope == "all-switch-ports":
        return n1_is_sw, n2_is_sw

    # default: switch-switch
    if n1_is_sw and n2_is_sw:
        return True, True
    return False, False


def ovs_apply_qos_linux_htb_fast(port_name, cap_mbps, qos_cfg, retries=8, retry_sleep_s=0.25):
    """
    Aplica QoS linux-htb con N colas usando UNA sola transacción ovs-vsctl por puerto.
    Sin tc/pfifo (limit_pkts ignorado) para máxima estabilidad en VM.
    """
    qos_type = str(qos_cfg.get("type", "linux-htb"))
    if qos_type != "linux-htb":
        raise ValueError("Only linux-htb supported (got {})".format(qos_type))

    queues = list(qos_cfg.get("queues", []))
    if not queues:
        raise ValueError("qos.queues is empty. Define at least one queue")

    default_qid = int(qos_cfg.get("default_queue_id", 0))

    ids = [int(q["id"]) for q in queues]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate queue ids in qos.queues: {}".format(ids))
    if default_qid not in set(ids):
        raise ValueError("default_queue_id={} not present in qos.queues ids={}".format(default_qid, ids))

    cap_bps = int(float(cap_mbps) * 1000000.0)

    # Validación suave: sum(min_ratio) <= 1
    sum_min = 0.0
    for q in queues:
        if q.get("min_ratio") is not None:
            sum_min += float(q.get("min_ratio"))
    if sum_min > 1.000001:
        raise ValueError("Sum of min_ratio exceeds 1.0 ({:.3f})".format(sum_min))

    ovs_timeout = int(qos_cfg.get("ovs_timeout_s", 5))
    use_no_wait = bool(qos_cfg.get("use_no_wait", True))

    base_cmd = SUDO + ["ovs-vsctl", "--timeout={}".format(ovs_timeout)]
    if use_no_wait:
        base_cmd.append("--no-wait")

    # 1 sola transacción: clear + set Port qos + create QoS + create Queue(s)
    cmd = base_cmd + [
        "--", "--if-exists", "clear", "Port", port_name, "qos",
        "--", "set", "Port", port_name, "qos=@qos",
        "--", "--id=@qos", "create", "QoS",
        "type={}".format(qos_type),
        "other-config:max-rate={}".format(cap_bps),
    ]

    for qid in ids:
        cmd.append("queues:{}=@q{}".format(qid, qid))

    for q in queues:
        qid = int(q["id"])
        min_bps = _bps_from_ratio(cap_mbps, q.get("min_ratio"))
        max_bps = _bps_from_ratio(cap_mbps, q.get("max_ratio"))

        if max_bps is None:
            max_bps = cap_bps
        max_bps = max(1, min(int(max_bps), cap_bps))
        if min_bps is not None:
            min_bps = max(1, min(int(min_bps), max_bps))

        cmd += ["--", "--id=@q{}".format(qid), "create", "Queue"]
        if min_bps is not None and min_bps > 0:
            cmd.append("other-config:min-rate={}".format(min_bps))
        cmd.append("other-config:max-rate={}".format(max_bps))

    last_err = None
    for _ in range(retries):
        try:
            run_cmd(cmd)
            return
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep_s)

    raise RuntimeError("[QOS] Failed to apply QoS on port={} after retries: {}".format(port_name, last_err))


def write_ready_marker():
    """
    Crea un marker para que tu bash pueda esperar a que termine QoS antes de iniciar capturas.
    Actívalo con env:
      - READY_MARKER_PATH=/tmp/sim_ready
    """
    p = os.environ.get("READY_MARKER_PATH", "").strip()
    if not p:
        return
    try:
        Path(p).write_text("ready\n")
    except Exception:
        pass


def main():
    cfg = load_topology()

    hosts_cfg = cfg.get("hosts", {})
    links_cfg = cfg.get("links", [])
    n_switches = int(cfg.get("n_switches", 0))

    qos_cfg = dict(cfg.get("qos", {}) or {})
    qos_enabled = bool(qos_cfg.get("enabled", False))

    ports_to_qos = []  # List[Tuple[str, float]]

    info("*** Creating network\n")
    cleanup()
    net = Containernet(controller=RemoteController, switch=OVSKernelSwitch)

    protocolName = "OpenFlow13"
    processes = {}
    client_threads = {}

    info("*** Adding controller\n")
    c0 = net.addController("c0", controller=RemoteController, ip="localhost", port=6653)

    # ---------------------- HOSTS (containers) ----------------------
    info("*** Adding containers\n")
    hosts = {}
    for name, params in hosts_cfg.items():
        ip = params["ip"]
        image = params["image"]
        volumes = params.get("volumes")

        docker_args = {"name": name, "ip": ip, "dimage": image}
        if volumes:
            docker_args["volumes"] = normalize_volumes(volumes)

        host = net.addDocker(**docker_args)
        hosts[name] = host

    # ---------------------- SWITCHES ----------------------
    info("*** Adding switches\n")
    switches = {}
    for i in range(1, n_switches + 1):
        sw_name = "switch{}".format(i)
        switches[sw_name] = net.addSwitch(sw_name, protocols=protocolName)

    # ---------------------- LINKS ----------------------
    info("*** Creating links\n")
    for item in links_cfg:
        (link_id, link_str), = item.items()
        left, right, bw_str = [x.strip() for x in link_str.split(",")]
        node1_name, _ = left.split("-")
        node2_name, _ = right.split("-")
        bw = float(bw_str)

        node1 = hosts.get(node1_name, switches.get(node1_name))
        node2 = hosts.get(node2_name, switches.get(node2_name))
        if node1 is None or node2 is None:
            raise ValueError("[{}] Unknown node: {} or {}".format(link_id, node1_name, node2_name))

        net.addLink(
            node1, node2,
            cls=TCLink,
            bw=bw,
            intfName1=left,
            intfName2=right,
        )

        if qos_enabled:
            do_left, do_right = should_apply_qos(node1_name, node2_name, switches, qos_cfg)
            if do_left:
                ports_to_qos.append((left, bw))
            if do_right:
                ports_to_qos.append((right, bw))

    info("*** Building network\n")
    net.build()

    info("*** Starting controller and switches\n")
    c0.start()
    for sw in switches.values():
        sw.start([c0])

    # Sleep global pequeño (evita race en VM)
    post_start_sleep_s = float(qos_cfg.get("post_start_sleep_s", 1.0))
    if post_start_sleep_s > 0:
        time.sleep(post_start_sleep_s)

    # ---------------------- QoS ----------------------
    if qos_enabled:
        # dedup preservando orden
        seen = set()
        dedup = []
        for (p, cap) in ports_to_qos:
            if p not in seen:
                dedup.append((p, cap))
                seen.add(p)
        ports_to_qos = dedup

        workers = int(qos_cfg.get("parallel_workers", 4))
        workers = max(1, min(workers, 8))
        strict = bool(qos_cfg.get("strict", False))

        info("*** [QOS] Applying QoS on {} ports (workers={})\n".format(len(ports_to_qos), workers))

        qos_errors = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for (port_name, cap) in ports_to_qos:
                fut = ex.submit(ovs_apply_qos_linux_htb_fast, port_name, cap, qos_cfg)
                futs[fut] = (port_name, cap)

            for fut in as_completed(futs):
                port_name, cap = futs[fut]
                try:
                    fut.result()
                except Exception as err:
                    msg = "*** [QOS][ERROR] port={} cap={}Mbps err={}\n".format(port_name, cap, err)
                    info(msg)
                    qos_errors.append((port_name, str(err)))

        if qos_errors:
            info("*** [QOS] Failed ports: {}\n".format(qos_errors))
            if strict:
                raise RuntimeError("QoS failed on some ports (strict=true)")

        info("*** [QOS] Done\n")

    # Marker opcional para tu bash
    write_ready_marker()

    # ---------------------- (Optional) Physical iface ----------------------
    # Actívalo con env:
    #   ADD_PHYS_IFACE=1 PHYS_SW=switch11 PHYS_IF=eno1
    if os.environ.get("ADD_PHYS_IFACE", "0") == "1":
        phys_sw = os.environ.get("PHYS_SW", "switch11")
        phys_if = os.environ.get("PHYS_IF", "eno1")
        if phys_sw in switches:
            info("*** Adding physical interface {} to {}\n".format(phys_if, phys_sw))
            switches[phys_sw].cmd("ovs-vsctl add-port {} {}".format(phys_sw, phys_if))

    # ---------------------- Default commands in containers ----------------------
    info("*** Setting default commands in containers\n")
    for name, params in hosts_cfg.items():
        host = hosts[name]

        mac = params.get("mac")
        if mac:
            host.setMAC(mac, intf="{}-eth0".format(name))

        intf = "{}-eth0".format(name)
        host.cmd("ip route del default || true")
        host.cmd("ip route add default dev {}".format(intf))

        cmd_default = params.get("comand_default")
        if not cmd_default:
            continue

        cmd_list = cmd_default if isinstance(cmd_default, list) else [cmd_default]
        for idx, cmd in enumerate(cmd_list):
            time.sleep(0.2)  # antes era 1s; en VM se siente
            proc_id = "{}#{}".format(name, idx) if len(cmd_list) > 1 else name
            print("[RUN] Running command on {}: {}".format(proc_id, cmd))
            process = host.popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            processes[proc_id] = process

    # pequeña pausa para que levanten servicios
    time.sleep(1.0)

    for pname, proc in processes.items():
        state = "V Active" if proc.poll() is None else "X Exited (code {})".format(proc.returncode)
        print("{}: {}".format(pname, state))

    # ---------------------- Burst traffic ----------------------
    time.sleep(2.0)
    info("*** Initializing Burst model\n")

    for name, params in hosts_cfg.items():
        host = hosts[name]
        flows_cfg = params.get("flows")
        if not flows_cfg:
            continue

        for flow_idx, flow in enumerate(flows_cfg):
            n_proc = int(flow.get("n_proc", 0) or 0)
            if n_proc <= 0:
                continue

            cmd_process = flow.get("comand_process")
            if not cmd_process:
                info("*** Flow missing 'comand_process' in {}: {}; skipping\n".format(
                    name, flow.get("name", flow_idx)
                ))
                continue

            conf = {
                "N_proc": n_proc,
                "cmd": cmd_process,
                "estados": flow.get("estados", []),
                "Q_matrix": np.array(flow.get("q_matrix", [])),
                "limits": [tuple(x) for x in flow.get("limits", [])],
            }

            flow_name = str(flow.get("name", "flow{}".format(flow_idx))).replace(" ", "_")

            for i in range(n_proc):
                unique_id = "{}_{}_{}".format(name, flow_name, i)
                t = Thread(target=run_burst, args=(host, unique_id, conf, processes))
                t.daemon = True
                t.start()
                client_threads[unique_id] = t

    # ---------------------- CLI / Non-interactive ----------------------
    if os.environ.get("NON_INTERACTIVE") == "1":
        info("*** Running in non-interactive mode; keeping network alive\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            info("*** Received interrupt, shutting down\n")
    else:
        CLI(net)


if __name__ == "__main__":
    net = None
    try:
        main()
    except Exception as e:
        info("\n\nError during simulation: {}\n\n".format(e))
        raise
    finally:
        info("*** Stopping network and cleaning up\n")
        try:
            # Containernet / Mininet tienen su propio stop
            # pero net es local en main(); cleanup() igual limpia
            pass
        finally:
            cleanup()
