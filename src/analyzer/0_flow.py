import yaml
import argparse
import subprocess
import hashlib
import tempfile
from pathlib import Path

import pandas as pd

# ========= helpers para topología =========

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_host_ip(hosts, name):
    ip_cidr = hosts[name]["ip"]
    return ip_cidr.split("/")[0]

def find_switch_iface_for_host(links, host_name):
    """
    Busca en 'links' una entrada tipo:
      "cliHttp1-eth0, switch12-eth3, 20"
    y devuelve "switch12-eth3".
    """
    for item in links:
        if isinstance(item, dict):
            for _lname, desc in item.items():
                parts = [p.strip() for p in desc.split(",")]
                if len(parts) < 2:
                    continue
                host_if, sw_if = parts[0], parts[1]
                if host_if.startswith(host_name + "-"):
                    return sw_if
    raise ValueError(f"No se encontró enlace para host {host_name}")

# ========= helpers para filtros y keys =========

def build_display_filter(protocol, transport, ip_src, ip_dst, src_port=None, dst_port=None, extra=""):
    """
    Construye el filtro para tshark. Los puertos se incluyen sólo
    si están presentes en la definición del flujo.
    """
    t = transport  # "tcp" o "udp"
    clauses = [f"ip.src=={ip_src}", f"ip.dst=={ip_dst}"]

    if protocol == "coap":
        proto_clause = "coap"
        port_prefix = "udp"
    elif protocol == "rtp":
        proto_clause = "rtp"
        port_prefix = t
    elif protocol:
        proto_clause = t
        port_prefix = t
    else:
        proto_clause = None
        port_prefix = t

    if proto_clause:
        clauses.append(proto_clause)

    if src_port is not None:
        clauses.append(f"{port_prefix}.srcport=={src_port}")
    if dst_port is not None:
        clauses.append(f"{port_prefix}.dstport=={dst_port}")

    base = " && ".join(clauses)
    if extra:
        base = f"({base}) && ({extra})"
    return base

def build_key_fields(protocol, transport):
    t = transport  # "tcp" o "udp"
    base = ["ip.src", "ip.dst", f"{t}.srcport", f"{t}.dstport"]

    if protocol == "http":
        return base + ["tcp.seq", "frame.len"]

    elif protocol == "coap":
        return base + ["coap.mid", "coap.token", "frame.len"]

    elif protocol == "rtp":
        # RTP sobre UDP: queremos diferenciar "ciclos" idénticos usando checksums
        if t == "udp":
            return [
                "ip.src",
                "ip.dst",
                "ip.checksum",   
                "udp.srcport",
                "udp.dstport",
                "rtp.ssrc",
                "rtp.seq",
            ]
        else:
            # por si algún día tienes RTP sobre TCP
            return [
                "ip.src",
                "ip.dst",
                "tcp.srcport",
                "tcp.dstport",
                "rtp.ssrc",
                "rtp.seq",
            ]


    elif protocol == "mqtt":
        return base + ["tcp.seq", "frame.len"]

    elif protocol == "udp":
        return base + ["udp.length", "udp.stream", "udp.checksum", "data.data"]

    else:
        return base + ["frame.len"]



def build_key(df, key_fields):
    """
    Construye una clave hash a partir de los campos indicados para
    poder emparejar paquetes observados en distintos puntos.
    """
    if df.empty:
        return pd.Series(dtype=str)

    for field in key_fields:
        if field not in df.columns:
            df[field] = ""

    concatenated = df[key_fields].astype(str).agg("|".join, axis=1)
    return concatenated.apply(lambda x: hashlib.sha256(x.encode()).hexdigest())


def run_tshark_to_csv(pcap_path, display_filter, fields, output_csv):
    """
    Ejecuta tshark para extraer sólo los campos necesarios a CSV.
    """
    tshark_fields = ["frame.time_epoch"] + list(dict.fromkeys(fields + ["frame.len"]))

    cmd = [
        "tshark", "-r", str(pcap_path),
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f",
    ]

    if display_filter:
        cmd += ["-Y", display_filter]

    for field in tshark_fields:
        cmd += ["-e", field]

    with open(output_csv, "w") as fh:
        subprocess.run(cmd, stdout=fh, check=True)


def analyze_flow(flow, output_dir):
    name = flow["name"]
    print(f"\n=== Analizando flujo: {name} ===")

    pcap_a = Path(flow["pcap_a"])
    pcap_b = Path(flow["pcap_b"])
    display_filter = flow.get("display_filter", "")
    key_fields = flow["key_fields"]
    jitter_enabled = flow.get("jitter", False)

    with tempfile.TemporaryDirectory(prefix=f"flow_{name}_") as tmpdir:
        tmp_a = Path(tmpdir) / "side_a.csv"
        tmp_b = Path(tmpdir) / "side_b.csv"

        run_tshark_to_csv(pcap_a, display_filter, key_fields, tmp_a)
        run_tshark_to_csv(pcap_b, display_filter, key_fields, tmp_b)

        try:
            df_a = pd.read_csv(tmp_a, on_bad_lines="skip", engine="python")
            df_b = pd.read_csv(tmp_b, on_bad_lines="skip", engine="python")
        except pd.errors.EmptyDataError:
            print("   ⚠️  Sin datos en uno de los PCAP; se omite.")
            return

        if df_a.empty or df_b.empty:
            print("   ⚠️  No se encontraron paquetes que cumplan el filtro.")
            return

    df_a = df_a.rename(columns={"frame.time_epoch": "t_a"})
    df_b = df_b.rename(columns={"frame.time_epoch": "t_b"})

    df_a = df_a.sort_values("t_a").reset_index(drop=True)
    df_b = df_b.sort_values("t_b").reset_index(drop=True)


    if "frame.len" not in df_a:
        df_a["frame.len"] = 0
    if "frame.len" not in df_b:
        df_b["frame.len"] = 0

    # 1) construir hash de key (con los campos de RTP que definimos antes)
    df_a["key"] = build_key(df_a, key_fields)
    df_b["key"] = build_key(df_b, key_fields)


    # Opcional, pero muy útil para debug
    if flow["protocol"] == "rtp":
        dups_a = df_a["key"].duplicated().any()
        dups_b = df_b["key"].duplicated().any()
        if dups_a or dups_b:
            print(f"   ⚠️ RTP: hay claves hash duplicadas. "
            f"A_dup={dups_a}, B_dup={dups_b}")

    # 2) índice dentro de cada key (0,1,2,...)
    df_a["idx"] = df_a.groupby("key").cumcount()
    df_b["idx"] = df_b.groupby("key").cumcount()

    # 3) match 1-a-1 por (key, idx)
    merged = df_a[["key", "idx", "t_a"]].merge(
        df_b[["key", "idx", "t_b"]],
        on=["key", "idx"],
        how="inner",
    )

    if merged.empty:
        print("   ⚠️  No se encontraron pares coincidentes entre ambas capturas.")
        return

    merged["delay"] = merged["t_b"] - merged["t_a"]
    merged["t_sec"] = merged["t_a"].astype(int)

    jitter_stats = None
    if jitter_enabled:
        merged_sorted = merged.sort_values("t_a").reset_index(drop=True)
        jitters = []
        prev_j = 0.0
        prev_delay = None
        for d in merged_sorted["delay"]:
            if prev_delay is None:
                prev_delay = d
                jitters.append(0.0)
                continue
            diff = abs(d - prev_delay)
            prev_j = prev_j + (diff - prev_j) / 16.0
            jitters.append(prev_j)
            prev_delay = d
        merged_sorted["jitter"] = jitters
        jitter_stats = (
            merged_sorted.groupby(merged_sorted["t_a"].astype(int))["jitter"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(columns={"t_a": "t_sec"})
        )

    delay_stats = (
        merged.groupby("t_sec")["delay"]
              .agg(["count", "mean", "std", "min", "max"])
              .reset_index()
    )

    delay_stats.insert(0, "flow", name)
    delay_stats["protocol"] = flow["protocol"]
    delay_stats["transport"] = flow["transport"]
    delay_stats["host_src"] = flow["host_src"]
    delay_stats["host_dst"] = flow["host_dst"]

    columns = [
        "flow", "protocol", "transport", "host_src", "host_dst",
        "t_sec", "count", "mean", "std", "min", "max"
    ]
    delay_stats = delay_stats[columns]

    # Pérdidas por segundo
    df_a_counts = df_a.assign(t_sec=df_a["t_a"].astype(int)).groupby("t_sec")["key"].count()
    df_b_counts = df_b.assign(t_sec=df_b["t_b"].astype(int)).groupby("t_sec")["key"].count()
    loss_df = (
        pd.concat([df_a_counts, df_b_counts], axis=1, keys=["count_a", "count_b"])
        .fillna(0)
    )
    loss_df["lost"] = loss_df["count_a"] - loss_df["count_b"]
    loss_df = loss_df.reset_index()

    # Throughput en B por segundo (bytes)
    throughput = (
        df_b.assign(t_sec=df_b["t_b"].astype(int))
        .groupby("t_sec")["frame.len"]
        .sum()
        .reset_index()
        .rename(columns={"frame.len": "throughput_bytes"})
    )

    result = delay_stats.copy()
    result = result.merge(loss_df, on="t_sec", how="left")
    result = result.merge(throughput, on="t_sec", how="left")
    if jitter_stats is not None:
        jitter_stats = jitter_stats.rename(columns={
            "count": "jitter_count",
            "mean": "jitter_mean",
            "std": "jitter_std",
            "min": "jitter_min",
            "max": "jitter_max",
        })
        result = result.merge(jitter_stats, on="t_sec", how="left")

    result = result.fillna(0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{name}.csv"
    result.to_csv(output_file, index=False)
    print(f"   ✅ Resultados guardados en {output_file}")
    print(f"   📊 Retardo (s): count={delay_stats['count'].sum()} mean={delay_stats['mean'].mean():.6f} min={delay_stats['min'].min():.6f} max={delay_stats['max'].max():.6f}")
    if jitter_stats is not None:
        print(f"   📈 Jitter (s): mean={jitter_stats['jitter_mean'].mean():.6f} max={jitter_stats['jitter_max'].max():.6f}")
    print(f"   📉 Perdidas totales: {(loss_df['lost'].clip(lower=0)).sum()}")
    print(f"   🚚 Throughput B (bytes totales): {throughput['throughput_bytes'].sum()}")

# ========= resolver flujos mínimos =========

def resolve_flows(flows_cfg_path, topology_path=None, pcap_root=None):
    cfg = load_yaml(flows_cfg_path)
    topo_file = topology_path or cfg.get("topology_file", "topology_conf.yml")
    topo = load_yaml(topo_file)
    pcap_root_dir = Path(pcap_root or cfg.get("pcap_root", "."))

    hosts = topo["hosts"]
    links = topo["links"]

    resolved = []

    for flow in cfg["flows"]:
        name = flow["name"]
        protocol = flow.get("protocol")
        transport = flow["transport"]
        src_port = flow.get("src_port")
        dst_port = flow.get("dst_port")
        extra_filter = flow.get("extra_filter", "")

        # 1) sacar hosts desde "srvHttp1-cliHttp1"
        host_src, host_dst = flow["link"].split("-")

        ip_src = get_host_ip(hosts, host_src)
        ip_dst = get_host_ip(hosts, host_dst)

        # 2) buscar interfaces de switch donde capturas
        sw_if_src = find_switch_iface_for_host(links, host_src)
        sw_if_dst = find_switch_iface_for_host(links, host_dst)

        pcap_a = pcap_root_dir / f"{sw_if_src}.pcap"
        pcap_b = pcap_root_dir / f"{sw_if_dst}.pcap"

        # 3) construir display_filter y key_fields
        display_filter = build_display_filter(
            protocol, transport, ip_src, ip_dst, src_port, dst_port, extra_filter
        )
        key_fields = build_key_fields(protocol, transport)


        resolved.append({
            "name": name,
            "protocol": protocol,
            "transport": transport,
            "host_src": host_src,
            "host_dst": host_dst,
            "ip_src": ip_src,
            "ip_dst": ip_dst,
            "src_port": src_port,
            "dst_port": dst_port,
            "pcap_a": str(pcap_a),
            "pcap_b": str(pcap_b),
            "display_filter": display_filter,
            "key_fields": key_fields,
            "jitter": flow.get("jitter", False),
        })

    return resolved

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resolver flujos y calcular métricas de retardo a partir de PCAPs."
    )
    parser.add_argument(
        "--flows",
        "-f",
        default="flows.yml",
        help="Ruta al YAML de flujos (por defecto: flows.yml)",
    )
    parser.add_argument(
        "--topology",
        "-t",
        default=None,
        help="Ruta al YAML de topología; si no se indica se usa la definida en el YAML de flujos",
    )
    parser.add_argument(
        "--pcaps",
        "-p",
        default=None,
        help="Directorio raíz donde se ubican los PCAP; si no se indica se usa el valor del YAML de flujos",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="flow_reports",
        help="Directorio donde se guardará un CSV por flujo (por defecto: flow_reports)",
    )
    args = parser.parse_args()

    flows = resolve_flows(args.flows, topology_path=args.topology, pcap_root=args.pcaps)
    if not flows:
        print("No se encontraron flujos para procesar.")
    else:
        for flow in flows:
            analyze_flow(flow, args.output_dir)
