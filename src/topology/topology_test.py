#!/usr/bin/python
from mininet.net import Containernet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.clean import cleanup

from burst_mod import run_burst
from subprocess import PIPE
from threading import Thread
import subprocess
import time
import yaml
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

setLogLevel('info')
TOPOLOGY_FILE = os.environ.get("TOPOLOGY_FILE", "topology_conf.yml")
BASE_DIR = Path(TOPOLOGY_FILE).resolve().parent

def run_cmd(cmd):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed rc={p.returncode}\ncmd={' '.join(cmd)}\nstdout={p.stdout}\nstderr={p.stderr}")
    return p

def _bps_from_ratio(cap_mbps: float, ratio: Optional[float]) -> Optional[int]:
    if ratio is None:
        return None
    return int(cap_mbps * float(ratio) * 1_000_000)

def ovs_apply_qos_linux_htb(port_name: str, cap_mbps: float, qos_cfg: Dict[str, Any]):
    """
    Aplica QoS linux-htb en 'port_name' con N colas definidas en qos_cfg['queues'].

    Requiere qos_cfg:
      type: "linux-htb"
      default_queue_id: int
      queues: [ {id, min_ratio?, max_ratio?, limit_pkts? , name? }, ... ]

    Buffer por cola:
      Se aplica con tc qdisc pfifo limit sobre parent 1:(queue_id+1).
    """
    wait_for_port(port_name, timeout_s=5.0)

    qos_type = str(qos_cfg.get("type", "linux-htb"))
    if qos_type != "linux-htb":
        raise ValueError(f"Only linux-htb supported by this helper (got {qos_type})")

    queues: List[Dict[str, Any]] = list(qos_cfg.get("queues", []))
    if not queues:
        raise ValueError("qos.queues is empty. Define at least one queue")

    # Validaciones básicas
    ids = [int(q["id"]) for q in queues]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate queue ids in qos.queues: {ids}")

    default_qid = int(qos_cfg.get("default_queue_id", 0))
    if default_qid not in set(ids):
        raise ValueError(f"default_queue_id={default_qid} not present in qos.queues ids={ids}")

    # Construir rates
    cap_bps = int(cap_mbps * 1_000_000)

    # Chequeo: suma de mínimos no debe exceder 100% (si está configurado)
    sum_min = 0.0
    for q in queues:
        if "min_ratio" in q and q["min_ratio"] is not None:
            sum_min += float(q["min_ratio"])
    if sum_min > 1.000001:
        raise ValueError(f"Sum of min_ratio exceeds 1.0 ({sum_min:.3f}). Reduce guarantees.")

    # 1) Limpia QoS previo
    run_cmd(["sudo", "ovs-vsctl", "--if-exists", "clear", "Port", port_name, "qos"])

    # 2) Arma comando ovs-vsctl para QoS + N queues
    cmd: List[str] = [
        "sudo", "ovs-vsctl",
        "--", "set", "Port", port_name, "qos=@qos",
        "--", "--id=@qos", "create", "QoS",
        f"type={qos_type}",
        f"other-config:max-rate={cap_bps}",
        f"other-config:default-queue={default_qid}",
    ]

    # mapping queues:<id>=@q<id>
    for qid in ids:
        cmd.append(f"queues:{qid}=@q{qid}")

    # define each queue object
    for q in queues:
        qid = int(q["id"])
        min_bps = _bps_from_ratio(cap_mbps, q.get("min_ratio"))
        max_bps = _bps_from_ratio(cap_mbps, q.get("max_ratio"))

        # Defaults razonables
        if max_bps is None:
            max_bps = cap_bps
        max_bps = max(1, min(max_bps, cap_bps))

        if min_bps is not None:
            min_bps = max(1, min(min_bps, max_bps))

        cmd += ["--", f"--id=@q{qid}", "create", "Queue"]
        # OVS linux-htb entiende min-rate/max-rate en other-config
        if min_bps is not None and min_bps > 0:
            cmd.append(f"other-config:min-rate={min_bps}")
        cmd.append(f"other-config:max-rate={max_bps}")

    run_cmd(cmd)

    # 3) Tamaño de buffer por cola (pfifo limit en paquetes)
    #    OVS suele mapear queue_id -> clase HTB minor = queue_id+1 => parent 1:(qid+1)
    #    Usamos 'replace' para que sea idempotente.
    for q in queues:
        qid = int(q["id"])
        limit_pkts = q.get("limit_pkts", None)
        if limit_pkts is None:
            continue

        parent = f"1:{qid + 1}"
        handle = f"{10 + qid}:"  # solo para que sea único
        run_cmd([
            "sudo", "tc", "qdisc", "replace",
            "dev", port_name,
            "parent", parent,
            "handle", handle,
            "pfifo", "limit", str(int(limit_pkts))
        ])
        
def ovs_port_exists(port_name: str) -> bool:
    r = subprocess.run(
        ["sudo", "ovs-vsctl", "--if-exists", "get", "Port", port_name, "name"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    return r.returncode == 0 and r.stdout.strip() != ""

def wait_for_port(port_name: str, timeout_s: float = 5.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if ovs_port_exists(port_name):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Port not found in OVSDB after {timeout_s}s: {port_name}")

def normalize_volumes(vol_list):
    resolved = []
    for v in vol_list:
        host, cont, *mode = v.split(":")
        host_path = Path(host)
        if not host_path.is_absolute():
            host_path = (BASE_DIR / host_path).resolve()

        if str(host_path).startswith("/sim/"):
            host_root_env = os.environ.get("HOST_PROJECT_ROOT")
            if not host_root_env:
                raise ValueError("HOST_PROJECT_ROOT must be set to remap /sim paths for Docker volumes")
            host_root = Path(host_root_env).resolve()
            host_path = host_root / host_path.relative_to("/sim")

        resolved.append(":".join([str(host_path), cont] + mode))
    return resolved

def load_topology(path=TOPOLOGY_FILE):
    with open(path, "r") as f:
        return yaml.safe_load(f)

try:
    cfg = load_topology()

    hosts_cfg = cfg.get("hosts", {})
    links_cfg = cfg.get("links", [])
    n_switches = cfg.get("n_switches", 0)
    ports_to_qos = []  # [(port_name, cap_mbps), ...]

    info('*** Creating network\n')
    cleanup()
    net = Containernet(controller=RemoteController, switch=OVSKernelSwitch)
    protocolName = "OpenFlow13"
    processes = {}
    client_threads = {}

    info('*** Adding controller\n')
    c0 = net.addController('c0', controller=RemoteController, ip='localhost', port=6653)

    # ---------------------- HOSTS (containers) ----------------------
    info('*** Adding containers\n')
    hosts = {}
    for name, params in hosts_cfg.items():
        ip = params["ip"]
        image = params["image"]
        volumes = params.get("volumes")

        if volumes:
            volumes = normalize_volumes(volumes)

        docker_args = {
            "name": name,
            "ip": ip,
            "dimage": image
        }

        if volumes:
            docker_args["volumes"] = volumes

        host = net.addDocker(**docker_args)
        hosts[name] = host

    # ---------------------- SWITCHES ----------------------
    info('*** Adding switches\n')
    switches = {}
    for i in range(1, n_switches + 1):
        sw_name = f"switch{i}"
        switches[sw_name] = net.addSwitch(sw_name, protocols=protocolName)

    # ---------------------- LINKS (switch-switch and host-switch) ----------------------
    info('*** Creating links\n')
    for item in links_cfg:
        # each item looks like: { "link0": "switch1-eth1, switch2-eth1, 30" }
        (link_id, link_str), = item.items()

        left, right, bw_str = [x.strip() for x in link_str.split(",")]
        node1_name, intf1_name = left.split("-")
        node2_name, intf2_name = right.split("-")
        bw = float(bw_str)

        # decide whether each node is a host or switch
        node1 = hosts.get(node1_name, switches.get(node1_name))
        node2 = hosts.get(node2_name, switches.get(node2_name))

        if node1 is None or node2 is None:
            raise ValueError(f"[{link_id}] Unknown node: {node1_name} or {node2_name}")
        

        # create the link with explicit interface names
        net.addLink(
            node1, node2,
            cls=TCLink,
            bw=bw,
            intfName1=left,
            intfName2=right,
        )

        qos_cfg = cfg.get("qos", {})
        qos_enabled = qos_cfg.get("enabled", False)
        crit_ratio = float(qos_cfg.get("link_crit_ratio", 0.3))

        if qos_enabled and node1_name in switches:
            ports_to_qos.append((left, bw))
        if qos_enabled and node2_name in switches:
            ports_to_qos.append((right, bw))

    info('*** Building network\n')
    net.build()


    info('*** Starting controller and switches\n')
    c0.start()
    for sw in switches.values():
        sw.start([c0])

    time.sleep(1)

    if qos_enabled:
        qos_errors = []
        for port_name, cap in ports_to_qos:
            try:
                info(f"*** [QOS] Applying queues port={port_name} cap={cap}Mbps crit_ratio={crit_ratio}\n")
                ovs_apply_qos_linux_htb(port_name, cap, qos_cfg)
                info(f"*** [QOS] Applied queues port={port_name}\n")
            except Exception as err:
                info(f"*** [QOS][ERROR] port={port_name} err={err}\n")
                qos_errors.append((port_name, str(err)))

        if qos_errors:
            info(f"*** [QOS] Failed on ports: {qos_errors}\n")

    info('*** (Optional) Connecting physical interface eno1 to switch11\n')
    # switches['switch3'].cmd('ovs-vsctl add-port switch3 eno1')
    #switches['switch11'].cmd('ovs-vsctl add-port switch11 eno1')

    # ---------------------- Default commands in containers ----------------------
    info('*** Setting default commands in containers\n')

    for name, params in hosts_cfg.items():
        host = hosts[name]

        # optional MAC override
        mac = params.get("mac")
        if mac:
            # assume the primary interface is name-eth0
            host.setMAC(mac, intf=f"{name}-eth0")


        intf = f"{name}-eth0"
        host.cmd("ip route del default || true")
        host.cmd(f"ip route add default dev {intf}")

        cmd_default = params.get("comand_default")
        if not cmd_default:
            continue  # comand_default is optional; skip if missing

        cmd_list = cmd_default if isinstance(cmd_default, list) else [cmd_default]
        for idx, cmd in enumerate(cmd_list):
            time.sleep(1)
            proc_id = f"{name}#{idx}" if len(cmd_list) > 1 else name
            print(f"[RUN] Running command on {proc_id}: {cmd}")
            process = host.popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            processes[proc_id] = process

    # small pause so services can start
    time.sleep(1)

    # check status of each process launched with comand_default
    for name, proc in processes.items():
        state = "V Active" if proc.poll() is None else f"X Exited (code {proc.returncode})"
        print(f"{name}: {state}")

    # ---------------------- Burst traffic using n_proc / q_matrix / limits ----------------------
    time.sleep(3)
    info('*** Initializing Burst model\n')

    for name, params in hosts_cfg.items():
        host = hosts[name]

        flows_cfg = params.get("flows")
        if not flows_cfg:
            continue 

        for flow_idx, flow in enumerate(flows_cfg):
            n_proc = flow.get("n_proc", 0)
            if not n_proc or n_proc <= 0:
                continue  # skip flows with non-positive process count

            cmd_process = flow.get("comand_process")
            if not cmd_process:
                info(f"*** Flow missing 'comand_process' in {name}: {flow.get('name', flow_idx)}; skipping\n")
                continue

            conf = {
                "N_proc": n_proc,
                "cmd": cmd_process,
                "estados": flow.get("estados", []),
                "Q_matrix": np.array(flow.get("q_matrix", [])),
                "limits": [tuple(x) for x in flow.get("limits", [])],
            }

            flow_name = str(flow.get("name", f"flow{flow_idx}")).replace(" ", "_")

            for i in range(n_proc):
                unique_id = f"{name}_{flow_name}_{i}"
                thread = Thread(target=run_burst, args=(host, unique_id, conf, processes))
                thread.start()
                client_threads[unique_id] = thread

    # ---------------------- CLI ----------------------
    if os.environ.get("NON_INTERACTIVE") == "1":
        info('*** Running in non-interactive mode; keeping network alive\n')
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            info('*** Received interrupt, shutting down non-interactive run\n')
    else:
        CLI(net)

except Exception as e:
    info(f"\n\nError during simulation: {e}\n\n")

finally:
    info('*** Stopping network and cleaning up\n')
    try:
        if 'net' in locals():
            net.stop()
    finally:
        cleanup()
