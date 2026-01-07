#!/usr/bin/python3
# -*- coding: utf-8 -*-

from mininet.net import Containernet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.clean import cleanup

from burst_mod import run_burst

import os
import time
import yaml
import subprocess
import traceback
import numpy as np

from pathlib import Path
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed


setLogLevel(os.environ.get("MN_LOGLEVEL", "warning"))

TOPOLOGY_FILE = os.environ.get("TOPOLOGY_FILE", "topology_conf.yml")
BASE_DIR = Path(TOPOLOGY_FILE).resolve().parent

IS_ROOT = (os.geteuid() == 0)
SUDO = [] if IS_ROOT else ["sudo"]


def run_cmd(cmd, check=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if check and p.returncode != 0:
        raise RuntimeError("cmd failed rc={}\ncmd={}\nstdout={}\nstderr={}".format(
            p.returncode, " ".join(cmd), p.stdout, p.stderr
        ))
    return p


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


def load_topology(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Topology YAML must be a dict at top-level")
    return cfg


def _bps(cap_mbps, ratio):
    if ratio is None:
        return None
    return int(float(cap_mbps) * float(ratio) * 1_000_000)


def ovs_apply_qos_linux_htb(port_name, cap_mbps, qos_cfg):
    qos_type = str(qos_cfg.get("type", "linux-htb"))
    if qos_type != "linux-htb":
        raise ValueError("Only linux-htb supported (got {})".format(qos_type))

    queues = list(qos_cfg.get("queues", []))
    if not queues:
        raise ValueError("qos.queues is empty")

    ids = [int(q["id"]) for q in queues]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate queue ids in qos.queues: {}".format(ids))

    default_qid = int(qos_cfg.get("default_queue_id", 0))
    if default_qid not in set(ids):
        raise ValueError("default_queue_id={} not present in qos.queues ids={}".format(default_qid, ids))

    cap_bps = int(float(cap_mbps) * 1_000_000)

    # clear previous qos (idempotent)
    run_cmd(SUDO + ["ovs-vsctl", "--if-exists", "clear", "Port", port_name, "qos"], check=False)

    cmd = []
    cmd += SUDO + [
        "ovs-vsctl",
        "--", "set", "Port", port_name, "qos=@qos",
        "--", "--id=@qos", "create", "QoS",
        "type={}".format(qos_type),
        "other-config:max-rate={}".format(cap_bps),
        "other-config:default-queue={}".format(default_qid),
    ]

    for qid in ids:
        cmd.append("queues:{}=@q{}".format(qid, qid))

    for q in queues:
        qid = int(q["id"])
        min_bps = _bps(cap_mbps, q.get("min_ratio"))
        max_bps = _bps(cap_mbps, q.get("max_ratio"))
        if max_bps is None:
            max_bps = cap_bps
        max_bps = max(1, min(int(max_bps), cap_bps))
        if min_bps is not None:
            min_bps = max(1, min(int(min_bps), max_bps))

        cmd += ["--", "--id=@q{}".format(qid), "create", "Queue"]
        if min_bps is not None:
            cmd.append("other-config:min-rate={}".format(min_bps))
        cmd.append("other-config:max-rate={}".format(max_bps))

    run_cmd(cmd, check=True)


def infer_ready_ports(cmd):
    c = str(cmd).lower()
    if "mosquitto" in c:
        return [1883]
    if "rabbit" in c:
        return [5672]
    if "mediamtx" in c:
        return [8554]
    if "vsftpd" in c:
        return [21]
    if "iperf" in c and " -s" in c:
        return [5001]
    if "web_server.py" in c:
        return [8080]
    return []


def host_has_python3(host):
    out = host.cmd("sh -lc 'command -v python3 >/dev/null 2>&1; echo $?'")
    return out.strip().endswith("0")


def tcp_check_localhost(host, port, timeout_s):
    if host_has_python3(host):
        cmd = (
            "python3 -c 'import socket; "
            "s=socket.socket(); s.settimeout({}); "
            "s.connect((\"127.0.0.1\", {})); s.close()' >/dev/null 2>&1; echo $?"
        ).format(float(timeout_s), int(port))
        rc = host.cmd("sh -lc \"{}\"".format(cmd)).strip()
        return rc.endswith("0")

    # fallback bash /dev/tcp
    cmd2 = "bash -lc 'exec 3<>/dev/tcp/127.0.0.1/{}' >/dev/null 2>&1; echo $?".format(int(port))
    rc2 = host.cmd("sh -lc \"{}\"".format(cmd2)).strip()
    return rc2.endswith("0")


def wait_ready(hosts, hosts_cfg, startup_cfg, names):
    strict = bool(startup_cfg.get("strict_ready", True))
    timeout_s = float(startup_cfg.get("ready_timeout_s", 45.0))
    interval_s = float(startup_cfg.get("ready_interval_s", 0.5))
    tcp_timeout_s = float(startup_cfg.get("ready_tcp_timeout_s", 0.4))
    workers = int(startup_cfg.get("ready_workers", 8))

    overrides = startup_cfg.get("ready_overrides", {}) or {}

    targets = []
    for name in names:
        cfg = hosts_cfg.get(name, {}) or {}

        ports = []
        if isinstance(overrides, dict) and name in overrides and isinstance(overrides[name], list):
            ports += [int(p) for p in overrides[name]]

        cmd_default = cfg.get("comand_default")
        if cmd_default:
            cmd_list = cmd_default if isinstance(cmd_default, list) else [cmd_default]
            for c in cmd_list:
                ports += infer_ready_ports(c)

        ports = sorted(list(set([p for p in ports if p and p > 0])))
        for p in ports:
            targets.append((name, p))

    if not targets:
        extra = float(startup_cfg.get("extra_wait_after_ready_s", 0.0))
        if extra > 0:
            time.sleep(extra)
        return

    pending = set(targets)
    t0 = time.time()

    while pending and (time.time() - t0) < timeout_s:
        done_now = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for (hn, port) in list(pending):
                futs[ex.submit(tcp_check_localhost, hosts[hn], port, tcp_timeout_s)] = (hn, port)
            for fut in as_completed(futs):
                hn, port = futs[fut]
                ok = False
                try:
                    ok = bool(fut.result())
                except Exception:
                    ok = False
                if ok:
                    done_now.append((hn, port))

        for x in done_now:
            if x in pending:
                pending.remove(x)

        if pending:
            time.sleep(interval_s)

    if pending and strict:
        raise RuntimeError("Some services not ready (TCP connect failed): {}".format(sorted(list(pending))))

    extra = float(startup_cfg.get("extra_wait_after_ready_s", 0.0))
    if extra > 0:
        time.sleep(extra)


def popen_service(host, cmd, service_stdio):
    """
    service_stdio:
      - "null":  no files, no docker logs spam  (stdout/stderr -> /dev/null)
      - "inherit": show in docker logs (still no files)
    """
    if service_stdio == "inherit":
        return host.popen(cmd, shell=True)

    devnull = open(os.devnull, "wb")
    try:
        return host.popen(cmd, shell=True, stdout=devnull, stderr=devnull)
    finally:
        devnull.close()


def classify_hosts(hosts_cfg, startup_cfg):
    srv_prefixes = startup_cfg.get("server_prefixes", ["srv", "brk"])
    cli_prefixes = startup_cfg.get("client_prefixes", ["cli", "emi"])

    servers, clients, others = [], [], []
    for name, cfg in hosts_cfg.items():
        role = str((cfg or {}).get("role", "")).lower().strip()
        if role == "server":
            servers.append(name); continue
        if role == "client":
            clients.append(name); continue

        low = name.lower()
        if any(low.startswith(p) for p in srv_prefixes):
            servers.append(name)
        elif any(low.startswith(p) for p in cli_prefixes):
            clients.append(name)
        else:
            others.append(name)
    return servers, clients, others


def start_commands_parallel(hosts, hosts_cfg, names, startup_cfg, processes):
    service_stdio = str(startup_cfg.get("service_stdio", "null")).lower()
    workers = int(startup_cfg.get("service_workers", 10))
    per_host_delay = float(startup_cfg.get("per_host_cmd_delay_s", 0.0))

    def _start_one(host_name):
        host = hosts[host_name]
        cfg = hosts_cfg.get(host_name, {}) or {}

        mac = cfg.get("mac")
        if mac:
            host.setMAC(mac, intf="{}-eth0".format(host_name))

        intf = "{}-eth0".format(host_name)
        host.cmd("ip route del default >/dev/null 2>&1 || true")
        host.cmd("ip route add default dev {} >/dev/null 2>&1 || true".format(intf))

        cmd_default = cfg.get("comand_default")
        if not cmd_default:
            return

        cmd_list = cmd_default if isinstance(cmd_default, list) else [cmd_default]
        for idx, cmd in enumerate(cmd_list):
            if per_host_delay > 0:
                time.sleep(per_host_delay)
            proc_id = "{}#{}".format(host_name, idx) if len(cmd_list) > 1 else host_name
            processes[proc_id] = popen_service(host, str(cmd), service_stdio)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_start_one, n) for n in names]
        for fut in as_completed(futs):
            fut.result()


def restart_exited_services(hosts, hosts_cfg, startup_cfg, processes):
    if not bool(startup_cfg.get("restart_exited_services", True)):
        return
    max_restart = int(startup_cfg.get("restart_max", 1))
    grace_s = float(startup_cfg.get("service_grace_s", 2.0))
    service_stdio = str(startup_cfg.get("service_stdio", "null")).lower()

    # map proc_id -> (host, cmd)
    proc_cmd = {}
    for host_name, cfg in hosts_cfg.items():
        cmd_default = (cfg or {}).get("comand_default")
        if not cmd_default:
            continue
        cmd_list = cmd_default if isinstance(cmd_default, list) else [cmd_default]
        for idx, cmd in enumerate(cmd_list):
            pid = "{}#{}".format(host_name, idx) if len(cmd_list) > 1 else host_name
            proc_cmd[pid] = (host_name, str(cmd))

    time.sleep(grace_s)

    for _ in range(max_restart):
        dead = []
        for pid, p in processes.items():
            if p is None:
                continue
            try:
                rc = p.poll()
            except Exception:
                rc = None
            if rc is not None:
                dead.append(pid)

        if not dead:
            return

        for pid in dead:
            if pid not in proc_cmd:
                continue
            host_name, cmd = proc_cmd[pid]
            processes[pid] = popen_service(hosts[host_name], cmd, service_stdio)

        time.sleep(grace_s)


def main():
    cfg = load_topology(TOPOLOGY_FILE)

    hosts_cfg = cfg.get("hosts", {}) or {}
    links_cfg = cfg.get("links", []) or []
    n_switches = int(cfg.get("n_switches", 0))

    qos_cfg = cfg.get("qos", {}) or {}
    qos_enabled = bool(qos_cfg.get("enabled", False))

    startup_cfg = cfg.get("startup", {}) or {}

    cleanup()
    net = Containernet(controller=RemoteController, switch=OVSKernelSwitch)

    c0 = net.addController("c0", controller=RemoteController, ip="localhost", port=6653)

    hosts = {}
    for name, params in hosts_cfg.items():
        ip = params["ip"]
        image = params["image"]
        volumes = params.get("volumes")
        if volumes:
            volumes = normalize_volumes(volumes)

        docker_args = {"name": name, "ip": ip, "dimage": image}
        if volumes:
            docker_args["volumes"] = volumes

        hosts[name] = net.addDocker(**docker_args)

    switches = {}
    protocolName = "OpenFlow13"
    for i in range(1, n_switches + 1):
        sw_name = "switch{}".format(i)
        switches[sw_name] = net.addSwitch(sw_name, protocols=protocolName)

    ports_to_qos = []
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

        net.addLink(node1, node2, cls=TCLink, bw=bw, intfName1=left, intfName2=right)

        if qos_enabled and node1_name in switches:
            ports_to_qos.append((left, bw))
        if qos_enabled and node2_name in switches:
            ports_to_qos.append((right, bw))

    net.build()
    c0.start()
    for sw in switches.values():
        sw.start([c0])

    time.sleep(float(startup_cfg.get("post_switch_start_sleep_s", 0.15)))

    if qos_enabled and ports_to_qos:
        seen = set()
        uniq = []
        for p, cap in ports_to_qos:
            if p not in seen:
                seen.add(p)
                uniq.append((p, cap))

        qos_workers = int(startup_cfg.get("qos_workers", 8))
        strict_qos = bool(startup_cfg.get("strict_qos", False))

        def _apply_one(port_cap):
            port_name, cap = port_cap
            ovs_apply_qos_linux_htb(port_name, cap, qos_cfg)

        errors = []
        with ThreadPoolExecutor(max_workers=qos_workers) as ex:
            futs = {ex.submit(_apply_one, pc): pc for pc in uniq}
            for fut in as_completed(futs):
                pc = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((pc[0], str(e)))

        if errors and strict_qos:
            raise RuntimeError("QoS failed on ports: {}".format(errors))

    servers, clients, others = classify_hosts(hosts_cfg, startup_cfg)

    processes = {}

    if servers:
        start_commands_parallel(hosts, hosts_cfg, servers, startup_cfg, processes)
        restart_exited_services(hosts, hosts_cfg, startup_cfg, processes)
        wait_ready(hosts, hosts_cfg, startup_cfg, servers)

    if others:
        start_commands_parallel(hosts, hosts_cfg, others, startup_cfg, processes)
        restart_exited_services(hosts, hosts_cfg, startup_cfg, processes)

    if clients:
        start_commands_parallel(hosts, hosts_cfg, clients, startup_cfg, processes)
        restart_exited_services(hosts, hosts_cfg, startup_cfg, processes)

    time.sleep(float(startup_cfg.get("pre_burst_sleep_s", 0.4)))

    client_threads = {}
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

    if os.environ.get("NON_INTERACTIVE") == "1":
        while True:
            time.sleep(1)
    else:
        CLI(net)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        try:
            cleanup()
        except Exception:
            pass
