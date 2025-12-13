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

setLogLevel('info')
TOPOLOGY_FILE = os.environ.get("TOPOLOGY_FILE", "topology_conf.yml")
BASE_DIR = Path(TOPOLOGY_FILE).resolve().parent

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
            "dimage": image,
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

    info('*** Building network\n')
    net.build()

    info('*** Starting controller and switches\n')
    c0.start()
    for sw in switches.values():
        sw.start([c0])

    info('*** (Optional) Connecting physical interface eno1 to switch11\n')
    # switches['switch3'].cmd('ovs-vsctl add-port switch3 eno1')
    switches['switch11'].cmd('ovs-vsctl add-port switch11 eno1')

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
