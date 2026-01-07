#!/usr/bin/python3
# Compatible con Python 3.6.9

from mininet.net import Containernet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.clean import cleanup

from burst_mod import run_burst

import os
import re
import time
import yaml
import numpy as np
import subprocess

from pathlib import Path
from subprocess import PIPE
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple, Set

setLogLevel("info")

TOPOLOGY_FILE = os.environ.get("TOPOLOGY_FILE", "topology_conf1_v2.yml")
BASE_DIR = Path(TOPOLOGY_FILE).resolve().parent

IS_ROOT = (os.geteuid() == 0)
SUDO = [] if IS_ROOT else ["sudo"]


# ----------------------------
# Helpers: filesystem / cmds
# ----------------------------

def load_topology(path=TOPOLOGY_FILE):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_volumes(vol_list):
    """
    Convierte volúmenes relativos al YAML a paths absolutos.
    Además remapea /sim/... usando HOST_PROJECT_ROOT.
    """
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


def write_ready_marker():
    """
    Marker opcional para que tu bash espere a que la topo termine QoS+servicios.
    Activa con env: READY_MARKER_PATH=/tmp/sim_ready
    """
    p = os.environ.get("READY_MARKER_PATH", "").strip()
    if not p:
        return
    try:
        Path(p).write_text("ready\n")
    except Exception:
        pass


# ----------------------------
# QoS: linux-htb (fast)
# ----------------------------

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


def ovs_apply_qos_linux_htb_fast(port_name, cap_mbps, qos_cfg, retries=10, retry_sleep_s=0.20):
    """
    Aplica QoS linux-htb con N colas en UNA transacción ovs-vsctl.
    Nota: NO toca tc/pfifo (limit_pkts), para estabilidad y velocidad en VM.
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

    # Validación suave sum(min_ratio) <= 1
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

    raise RuntimeError("[QOS] Failed port={} after retries: {}".format(port_name, last_err))


# ----------------------------
# Startup: services + readiness
# ----------------------------

_IP_RE = r"(?:\d{1,3}\.){3}\d{1,3}"
RE_HOST = re.compile(r"--host\s+({})".format(_IP_RE))
RE_PORT = re.compile(r"--port\s+(\d{2,5})\b")
RE_URL = re.compile(r"(?:http|https|ws|wss|rtsp)://({})(?::(\d{{2,5}}))?".format(_IP_RE))


def extract_targets_from_cmd(cmd):
    """
    Extrae (ip, port) de:
      - --host X.X.X.X --port N
      - URLs http://ip:port / ws://ip:port / rtsp://ip:port
    Retorna lista de tuples (ip, port_or_None)
    """
    out = []

    hosts = RE_HOST.findall(cmd) or []
    ports = RE_PORT.findall(cmd) or []

    # caso típico: --host ip --port p
    if hosts:
        if ports:
            # si hay varios ports, tomamos el primero (normalmente solo hay uno)
            out.append((hosts[0], int(ports[0])))
        else:
            out.append((hosts[0], None))

    # URLs
    for ip, p in RE_URL.findall(cmd):
        out.append((ip, int(p) if p else None))

    # dedup manteniendo orden
    seen = set()
    res = []
    for ip, p in out:
        key = (ip, p)
        if key not in seen:
            seen.add(key)
            res.append((ip, p))
    return res


def infer_default_port(flow_name, cmd):
    """
    Heurística para puertos por protocolo cuando no viene --port.
    Útil para MQTT/AMQP/FTP/RTSP/HTTP básicos.
    """
    s = (flow_name + " " + cmd).lower()

    # MQTT
    if "mqtt" in s or "mosquitto" in s or "paho" in s:
        return 1883

    # AMQP/RabbitMQ
    if "amqp" in s or "rabbit" in s or "pika" in s:
        return 5672

    # FTP
    if "ftp" in s or "vsftpd" in s:
        return 21

    # RTSP (si usas MediaMTX u otro)
    if "rtsp" in s:
        return 8554

    # HTTP/Web
    if "http" in s or "web" in s or "curl" in s:
        return 80

    return None


def host_popen_wait(host, cmd, timeout_s=2.0):
    """
    Ejecuta un comando dentro del host mininet/docker y espera retorno.
    Retorna (rc, out, err)
    """
    p = host.popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    try:
        out, err = p.communicate(timeout=timeout_s)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
        return 124, b"", b"timeout"
    return p.returncode, out, err


def tcp_ready_check(probe_host, ip, port, per_try_timeout=0.5):
    """
    Prueba conexión TCP desde probe_host hacia ip:port.
    Retorna True si conecta.
    """
    py = (
        "python3 - <<'PY'\n"
        "import socket, sys\n"
        "ip={!r}\n"
        "port={}\n"
        "s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout({})\n"
        "try:\n"
        "    s.connect((ip, port))\n"
        "    s.close()\n"
        "    sys.exit(0)\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
        "PY\n"
    ).format(ip, int(port), float(per_try_timeout))

    rc, _, _ = host_popen_wait(probe_host, py, timeout_s=per_try_timeout + 0.5)
    return rc == 0


def wait_for_targets(probe_host, targets, ready_timeout_s=60.0, interval_s=0.5, workers=6):
    """
    Espera a que TODOS los targets TCP estén arriba (con timeout global por target).
    targets: List[(ip,port)]
    """
    if not targets:
        return []

    def _wait_one(ip, port):
        t0 = time.time()
        while time.time() - t0 < ready_timeout_s:
            if tcp_ready_check(probe_host, ip, port, per_try_timeout=min(0.7, interval_s)):
                return (ip, port, True, time.time() - t0)
            time.sleep(interval_s)
        return (ip, port, False, ready_timeout_s)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_wait_one, ip, port) for (ip, port) in targets]
        for fut in as_completed(futs):
            results.append(fut.result())

    # ordena para log bonito
    results.sort(key=lambda x: (x[0], x[1]))
    return results


def build_ready_targets(cfg):
    """
    Construye targets desde flows[*].comand_process:
      - extrae ip/port
      - si falta port, usa heurística por flow_name/cmd
      - dedup
    Permite override manual:
      cfg.get("startup",{}).get("ready_targets",[]) = [{ip:..., port:...}, ...]
    """
    hosts_cfg = cfg.get("hosts", {}) or {}
    targets = []

    # targets desde flows
    for host_name, params in hosts_cfg.items():
        flows = params.get("flows") or []
        for flow in flows:
            flow_name = str(flow.get("name", ""))
            cmd = str(flow.get("comand_process", ""))
            if not cmd:
                continue
            for ip, port in extract_targets_from_cmd(cmd):
                if port is None:
                    port = infer_default_port(flow_name, cmd)
                if port is None:
                    continue
                targets.append((ip, int(port)))

    # manual ready targets
    startup_cfg = cfg.get("startup", {}) or {}
    manual = startup_cfg.get("ready_targets", []) or []
    for item in manual:
        try:
            ip = str(item.get("ip"))
            port = int(item.get("port"))
            targets.append((ip, port))
        except Exception:
            pass

    # dedup
    seen = set()
    out = []
    for ip, port in targets:
        key = (ip, int(port))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def start_services_parallel(hosts_cfg, hosts, processes, stdio_mode="files", logs_dir="/sim/results/logs/host_procs",
                            workers=8):
    """
    Arranca comand_default por host en paralelo:
      - Si comand_default es lista, la junta con '; ' en un solo bash -lc "...".
      - 1 popen por host (rápido).
    stdio_mode: files|null|inherit
    """
    try:
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    def _stdio(proc_id):
        if stdio_mode == "null":
            devnull = open(os.devnull, "w")
            return devnull, devnull
        if stdio_mode == "inherit":
            return None, None
        # files
        out_path = str(Path(logs_dir) / ("{}.out.log".format(proc_id)))
        err_path = str(Path(logs_dir) / ("{}.err.log".format(proc_id)))
        return open(out_path, "w"), open(err_path, "w")

    def _start_one(name):
        params = hosts_cfg[name]
        host = hosts[name]

        # MAC opcional
        mac = params.get("mac")
        if mac:
            host.setMAC(mac, intf="{}-eth0".format(name))

        # default route (rápido)
        intf = "{}-eth0".format(name)
        host.cmd("ip route del default || true; ip route add default dev {}".format(intf))

        cmd_default = params.get("comand_default")
        if not cmd_default:
            return (name, None, None)

        cmd_list = cmd_default if isinstance(cmd_default, list) else [cmd_default]
        joined = "; ".join([str(x) for x in cmd_list if str(x).strip() != ""])
        if not joined.strip():
            return (name, None, None)

        proc_id = name
        out_f, err_f = _stdio(proc_id)
        # Nota: NO cierres out_f/err_f aquí; los usa el proceso.
        p = host.popen("bash -lc {}".format(repr(joined)), shell=True, stdout=out_f, stderr=err_f)
        processes[proc_id] = p
        return (name, proc_id, joined)

    started = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_start_one, name): name for name in hosts_cfg.keys()}
        for fut in as_completed(futs):
            started.append(fut.result())

    return started


# ----------------------------
# Main
# ----------------------------

def main():
    cfg = load_topology()

    # Config opcional para acelerar o controlar “startup”
    startup_cfg = cfg.get("startup", {}) or {}
    READY_TIMEOUT_S = float(startup_cfg.get("ready_timeout_s", float(os.environ.get("READY_TIMEOUT_S", "60"))))
    READY_INTERVAL_S = float(startup_cfg.get("ready_interval_s", float(os.environ.get("READY_INTERVAL_S", "0.5"))))
    READY_WORKERS = int(startup_cfg.get("ready_parallel_workers", int(os.environ.get("READY_WORKERS", "6"))))
    STRICT_READY = bool(startup_cfg.get("strict_ready", True))  # por defecto: asegurar
    SERVICES_WORKERS = int(startup_cfg.get("service_workers", int(os.environ.get("SERVICE_WORKERS", "8"))))
    STDIO_MODE = str(startup_cfg.get("stdio_mode", os.environ.get("CMD_STDIO_MODE", "files"))).strip().lower()
    LOGS_DIR = str(startup_cfg.get("logs_dir", os.environ.get("HOST_PROC_LOG_DIR", "/sim/results/logs/host_procs")))

    SKIP_CLEANUP = (os.environ.get("SKIP_MININET_CLEANUP", "0") == "1")

    hosts_cfg = cfg.get("hosts", {}) or {}
    links_cfg = cfg.get("links", []) or []
    n_switches = int(cfg.get("n_switches", 0))

    qos_cfg = dict(cfg.get("qos", {}) or {})
    qos_enabled = bool(qos_cfg.get("enabled", False))

    ports_to_qos = []  # List[Tuple[str, float]]

    info("*** Creating network\n")
    if not SKIP_CLEANUP:
        cleanup()

    net = Containernet(controller=RemoteController, switch=OVSKernelSwitch)
    protocolName = "OpenFlow13"
    processes = {}
    client_threads = {}

    info("*** Adding controller\n")
    c0 = net.addController("c0", controller=RemoteController, ip="localhost", port=6653)

    # ---------------------- HOSTS ----------------------
    info("*** Adding containers (hosts={})\n".format(len(hosts_cfg)))
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
    info("*** Adding switches (n={})\n".format(n_switches))
    switches = {}
    for i in range(1, n_switches + 1):
        sw_name = "switch{}".format(i)
        switches[sw_name] = net.addSwitch(sw_name, protocols=protocolName)

    # ---------------------- LINKS ----------------------
    info("*** Creating links (n={})\n".format(len(links_cfg)))
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

    # Sleep global corto (evita carreras en VM)
    post_start_sleep_s = float(qos_cfg.get("post_start_sleep_s", 0.8))
    if post_start_sleep_s > 0:
        time.sleep(post_start_sleep_s)

    # ---------------------- QoS ----------------------
    if qos_enabled:
        # dedup ports
        seen = set()
        dedup = []
        for (p, cap) in ports_to_qos:
            if p not in seen:
                dedup.append((p, cap))
                seen.add(p)
        ports_to_qos = dedup

        workers = int(qos_cfg.get("parallel_workers", 4))
        workers = max(1, min(workers, 8))
        strict_qos = bool(qos_cfg.get("strict", False))

        info("*** [QOS] Applying QoS on {} ports (workers={})\n".format(len(ports_to_qos), workers))

        qos_errors = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for (port_name, cap) in ports_to_qos:
                futs[ex.submit(ovs_apply_qos_linux_htb_fast, port_name, cap, qos_cfg)] = port_name

            for fut in as_completed(futs):
                port_name = futs[fut]
                try:
                    fut.result()
                except Exception as err:
                    info("*** [QOS][ERROR] port={} err={}\n".format(port_name, err))
                    qos_errors.append((port_name, str(err)))

        if qos_errors:
            info("*** [QOS] Failed ports: {}\n".format(qos_errors))
            if strict_qos:
                raise RuntimeError("QoS failed on some ports (strict=true)")

        info("*** [QOS] Done\n")

    # ---------------------- OPTIONAL physical iface ----------------------
    # env: ADD_PHYS_IFACE=1 PHYS_SW=switch11 PHYS_IF=eno1
    if os.environ.get("ADD_PHYS_IFACE", "0") == "1":
        phys_sw = os.environ.get("PHYS_SW", "switch11")
        phys_if = os.environ.get("PHYS_IF", "eno1")
        if phys_sw in switches:
            info("*** Adding physical interface {} to {}\n".format(phys_if, phys_sw))
            switches[phys_sw].cmd("ovs-vsctl add-port {} {}".format(phys_sw, phys_if))

    # ---------------------- SERVICES START (PARALLEL) ----------------------
    info("*** Starting services (comand_default) in parallel (workers={})\n".format(SERVICES_WORKERS))
    started = start_services_parallel(
        hosts_cfg=hosts_cfg,
        hosts=hosts,
        processes=processes,
        stdio_mode=STDIO_MODE,
        logs_dir=LOGS_DIR,
        workers=SERVICES_WORKERS
    )

    # Log corto de qué se arrancó
    started_services = [x for x in started if x[1] is not None]
    info("*** Services launched: {}\n".format(len(started_services)))

    # ---------------------- READINESS BEFORE CLIENTS ----------------------
    # Elegimos probe host: el primero que tenga flows (cliente), para que tenga Python.
    probe_host = None
    for hname, params in hosts_cfg.items():
        if params.get("flows"):
            probe_host = hosts[hname]
            info("*** Probe host for readiness: {}\n".format(hname))
            break
    if probe_host is None:
        # fallback: cualquier host
        probe_host = next(iter(hosts.values()))
        info("*** Probe host for readiness: (fallback) {}\n".format(probe_host.name))

    targets = build_ready_targets(cfg)
    info("*** Readiness targets (TCP): {}\n".format(len(targets)))
    if targets:
        info("*** Waiting readiness: timeout={}s interval={}s workers={} strict={}\n".format(
            READY_TIMEOUT_S, READY_INTERVAL_S, READY_WORKERS, STRICT_READY
        ))
        results = wait_for_targets(
            probe_host=probe_host,
            targets=targets,
            ready_timeout_s=READY_TIMEOUT_S,
            interval_s=READY_INTERVAL_S,
            workers=READY_WORKERS
        )

        failed = [r for r in results if not r[2]]
        ok = [r for r in results if r[2]]

        info("*** Ready OK: {}\n".format(len(ok)))
        if failed:
            info("*** Ready FAILED: {}\n".format(len(failed)))
            for (ip, port, _, waited) in failed[:12]:
                info("    - {}:{} (waited {}s)\n".format(ip, port, int(waited)))
            if STRICT_READY:
                raise RuntimeError("Some services not ready; aborting because strict_ready=true")
        else:
            info("*** All readiness targets are up.\n")

    # marker opcional (para tu bash)
    write_ready_marker()

    # ---------------------- BURST / CLIENT TRAFFIC ----------------------
    info("*** Initializing Burst model (clients)\n")
    # pequeño sleep solo para estabilidad (no por host)
    time.sleep(float(startup_cfg.get("pre_clients_sleep_s", 0.2)))

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
    try:
        main()
    except Exception as e:
        info("\n\nError during simulation: {}\n\n".format(e))
        raise
    finally:
        info("*** Cleaning up\n")
        cleanup()
