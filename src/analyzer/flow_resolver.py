import argparse
import concurrent.futures
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml

# Factor usado para descartar retardos/jitters atípicos
OUTLIER_STD_FACTOR = 5

# ========= helpers estadísticos =========

def weighted_avg(values, weights):
    total_weight = weights.sum()
    if total_weight == 0:
        return 0
    return (values * weights).sum() / total_weight

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


def analyze_direction(flow, direction_info):
    """
    Analiza una dirección (A->B o B->A) y devuelve métricas y DataFrames.
    direction_info debe contener:
      direction, pcap_src, pcap_dst, ip_src, ip_dst, src_port, dst_port, host_src, host_dst
    """
    name = flow["name"]
    protocol = flow["protocol"]
    transport = flow["transport"]
    jitter_enabled = flow.get("jitter", False)
    extra_filter = flow.get("extra_filter", "")
    key_fields = flow["key_fields"]

    display_filter = build_display_filter(
        protocol,
        transport,
        direction_info["ip_src"],
        direction_info["ip_dst"],
        direction_info.get("src_port"),
        direction_info.get("dst_port"),
        extra_filter,
    )

    pcap_src = Path(direction_info["pcap_src"])
    pcap_dst = Path(direction_info["pcap_dst"])

    with tempfile.TemporaryDirectory(prefix=f"flow_{name}_{direction_info['direction']}_") as tmpdir:
        tmp_a = Path(tmpdir) / "side_a.csv"
        tmp_b = Path(tmpdir) / "side_b.csv"

        run_tshark_to_csv(pcap_src, display_filter, key_fields, tmp_a)
        run_tshark_to_csv(pcap_dst, display_filter, key_fields, tmp_b)

        try:
            df_a = pd.read_csv(tmp_a, on_bad_lines="skip", engine="python")
            df_b = pd.read_csv(tmp_b, on_bad_lines="skip", engine="python")
        except pd.errors.EmptyDataError:
            print(f"   ⚠️  {direction_info['direction']}: sin datos en uno de los PCAP; se omite.")
            return None

        if df_a.empty or df_b.empty:
            print(f"   ⚠️  {direction_info['direction']}: no se encontraron paquetes que cumplan el filtro.")
            return None

    df_a = df_a.rename(columns={"frame.time_epoch": "t_a"})
    df_b = df_b.rename(columns={"frame.time_epoch": "t_b"})

    df_a = df_a.sort_values("t_a").reset_index(drop=True)
    df_b = df_b.sort_values("t_b").reset_index(drop=True)

    if "frame.len" not in df_a:
        df_a["frame.len"] = 0
    if "frame.len" not in df_b:
        df_b["frame.len"] = 0

    # 1) construir hash de key
    df_a["key"] = build_key(df_a, key_fields)
    df_b["key"] = build_key(df_b, key_fields)

    if protocol == "rtp":
        dups_a = df_a["key"].duplicated().any()
        dups_b = df_b["key"].duplicated().any()
        if dups_a or dups_b:
            print(f"   ⚠️ {direction_info['direction']} RTP: hay claves hash duplicadas. A_dup={dups_a}, B_dup={dups_b}")

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
        print(f"   ⚠️  {direction_info['direction']}: no se encontraron pares coincidentes entre ambas capturas.")
        return None

    merged["delay"] = merged["t_b"] - merged["t_a"]
    merged["t_sec"] = merged["t_a"].astype(int)

    # Filtrar retardos negativos y outliers respecto a la media
    merged = merged[merged["delay"] >= 0].copy()
    if merged.empty:
        print(f"   ⚠️  {direction_info['direction']}: todos los retardos fueron negativos; se omite.")
        return None

    delay_mean = merged["delay"].mean()
    delay_std = merged["delay"].std()
    if pd.notna(delay_std) and delay_std > 0:
        delay_upper = delay_mean + OUTLIER_STD_FACTOR * delay_std
        merged = merged[merged["delay"] <= delay_upper].copy()
        if merged.empty:
            print(f"   ⚠️  {direction_info['direction']}: todos los retardos se descartaron como outliers; se omite.")
            return None

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

        # Filtrar jitters atípicos
        jitter_mean = merged_sorted["jitter"].mean()
        jitter_std = merged_sorted["jitter"].std()
        if pd.notna(jitter_std) and jitter_std > 0:
            jitter_upper = jitter_mean + OUTLIER_STD_FACTOR * jitter_std
            merged_sorted = merged_sorted[merged_sorted["jitter"] <= jitter_upper].copy()

        if not merged_sorted.empty:
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
    delay_stats["protocol"] = protocol
    delay_stats["transport"] = transport
    delay_stats["host_src"] = direction_info["host_src"]
    delay_stats["host_dst"] = direction_info["host_dst"]

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
    result["direction"] = direction_info["direction"]

    # Acumulados para sumas globales
    delay_count = delay_stats["count"].sum()
    delay_sum = (delay_stats["mean"] * delay_stats["count"]).sum()
    delay_min = delay_stats["min"].min()
    delay_max = delay_stats["max"].max()

    jitter_count = 0
    jitter_sum = 0
    jitter_max = 0
    if jitter_stats is not None and not jitter_stats.empty:
        jitter_count = jitter_stats["jitter_count"].sum()
        jitter_sum = (jitter_stats["jitter_mean"] * jitter_stats["jitter_count"]).sum()
        jitter_max = jitter_stats["jitter_max"].max()

    throughput_total = throughput["throughput_bytes"].sum()
    loss_total = loss_df["lost"].sum()

    return {
        "result": result,
        "loss_df": loss_df,
        "throughput_total": throughput_total,
        "loss_total": loss_total,
        "delay_count": delay_count,
        "delay_sum": delay_sum,
        "delay_min": delay_min,
        "delay_max": delay_max,
        "jitter_count": jitter_count,
        "jitter_sum": jitter_sum,
        "jitter_max": jitter_max,
        "direction": direction_info["direction"],
    }


def analyze_flow(flow, output_dir):
    name = flow["name"]
    print(f"\n=== Analizando flujo: {name} (bidireccional) ===")

    forward_info = {
        "direction": "forward",
        "pcap_src": flow["pcap_a"],
        "pcap_dst": flow["pcap_b"],
        "ip_src": flow["ip_src"],
        "ip_dst": flow["ip_dst"],
        "src_port": flow.get("src_port"),
        "dst_port": flow.get("dst_port"),
        "host_src": flow["host_src"],
        "host_dst": flow["host_dst"],
    }
    reverse_info = {
        "direction": "reverse",
        "pcap_src": flow["pcap_b"],
        "pcap_dst": flow["pcap_a"],
        "ip_src": flow["ip_dst"],
        "ip_dst": flow["ip_src"],
        "src_port": flow.get("dst_port"),
        "dst_port": flow.get("src_port"),
        "host_src": flow["host_dst"],
        "host_dst": flow["host_src"],
    }

    metrics = []
    for info in (forward_info, reverse_info):
        direction_metrics = analyze_direction(flow, info)
        if direction_metrics is not None:
            metrics.append(direction_metrics)

    if not metrics:
        print("   ⚠️  No se obtuvieron métricas en ningún sentido.")
        return

    # Combinar resultados por segundo y escribir CSV (sin columna direction, agregando ambos sentidos)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{name}.csv"
    combined_result = pd.concat([m["result"] for m in metrics], ignore_index=True).fillna(0)
    combined_result["endpoints"] = f"{flow['host_src']}<->{flow['host_dst']}"

    agg_rows = []
    for t_sec, grp in combined_result.groupby("t_sec"):
        row = {
            "flow": name,
            "protocol": flow["protocol"],
            "transport": flow["transport"],
            "endpoints": combined_result["endpoints"].iloc[0],
            "t_sec": t_sec,
        }
        row["count"] = grp["count"].sum()
        row["mean"] = weighted_avg(grp["mean"], grp["count"])
        row["std"] = weighted_avg(grp["std"], grp["count"])
        row["min"] = grp["min"].min()
        row["max"] = grp["max"].max()
        row["count_a"] = grp["count_a"].sum() if "count_a" in grp else 0
        row["count_b"] = grp["count_b"].sum() if "count_b" in grp else 0
        row["lost"] = grp["lost"].sum() if "lost" in grp else 0
        row["throughput_bytes"] = grp["throughput_bytes"].sum() if "throughput_bytes" in grp else 0
        if "jitter_count" in grp:
            row["jitter_count"] = grp["jitter_count"].sum()
            row["jitter_mean"] = weighted_avg(grp["jitter_mean"], grp["jitter_count"])
            row["jitter_std"] = weighted_avg(grp["jitter_std"], grp["jitter_count"])
            row["jitter_min"] = grp["jitter_min"].min()
            row["jitter_max"] = grp["jitter_max"].max()
        agg_rows.append(row)

    combined_result = pd.DataFrame(agg_rows).sort_values("t_sec")
    combined_result = combined_result.fillna(0)
    combined_result.to_csv(output_file, index=False)

    # Agregados totales
    total_delay_count = combined_result["count"].sum()
    delay_mean = weighted_avg(combined_result["mean"], combined_result["count"])
    delay_min = combined_result["min"].min() if not combined_result.empty else 0
    delay_max = combined_result["max"].max() if not combined_result.empty else 0

    total_lost = combined_result["lost"].sum() if "lost" in combined_result else 0
    total_lost = max(total_lost, 0)
    total_throughput = combined_result["throughput_bytes"].sum() if "throughput_bytes" in combined_result else 0

    total_jitter_count = combined_result["jitter_count"].sum() if "jitter_count" in combined_result else 0
    jitter_mean = weighted_avg(combined_result["jitter_mean"], combined_result["jitter_count"]) if "jitter_mean" in combined_result else 0
    jitter_max = combined_result["jitter_max"].max() if "jitter_max" in combined_result else 0

    print(f"   ✅ Resultados guardados en {output_file}")
    print(f"   📊 Retardo combinado (s): count={total_delay_count} mean={delay_mean:.6f} min={delay_min:.6f} max={delay_max:.6f}")
    if total_jitter_count:
        print(f"   📈 Jitter combinado (s): mean={jitter_mean:.6f} max={jitter_max:.6f}")
    print(f"   📉 Perdidas totales bidireccionales: {total_lost}")
    print(f"   🚚 Throughput B (bytes totales bidireccionales): {total_throughput}")

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
            "extra_filter": extra_filter,
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
        if len(flows) == 1:
            for flow in flows:
                analyze_flow(flow, args.output_dir)
        else:
            workers = min(len(flows), os.cpu_count() or len(flows))
            print(f"Ejecutando en paralelo con {workers} procesos...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                fut_map = {
                    executor.submit(analyze_flow, flow, args.output_dir): flow["name"]
                    for flow in flows
                }
                for fut in concurrent.futures.as_completed(fut_map):
                    name = fut_map[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"   ⚠️  Flujo {name} falló: {exc}")
