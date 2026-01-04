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
    Construye una clave hash rápida (uint64) a partir de los campos indicados para
    poder emparejar paquetes observados en distintos puntos.

    Antes se usaba sha256 por fila (muy costoso); aquí usamos hashing vectorizado de pandas.
    """
    if df.empty:
        return pd.Series(dtype="uint64")

    for field in key_fields:
        if field not in df.columns:
            df[field] = ""

    key_df = df[key_fields].copy()
    # Evita NaNs para que el hash sea consistente y más rápido
    key_df = key_df.fillna("").astype("string")
    return pd.util.hash_pandas_object(key_df, index=False)


def _tmp_base_dir():
    """
    Usa /dev/shm (tmpfs) si está disponible para que los temporales vivan en RAM.
    """
    shm = Path("/dev/shm")
    if shm.exists() and os.access(str(shm), os.W_OK):
        return str(shm)
    return None


def _port_prefix(protocol, transport):
    """
    Replica la lógica de build_display_filter para saber qué prefijo de puertos se usa.
    """
    if protocol == "coap":
        return "udp"
    return transport


def run_tshark_to_tsv(pcap_path, display_filter, fields, output_tsv):
    """
    Ejecuta tshark para extraer sólo los campos necesarios a TSV (más rápido de parsear que CSV con quotes).
    - Usa '-n' para evitar resolución de nombres.
    - Usa separador TAB y sin comillas.
    """
    tshark_fields = ["frame.time_epoch"] + list(dict.fromkeys(list(fields) + ["frame.len"]))

    cmd = [
        "tshark", "-n",
        "-r", str(pcap_path),
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=\t",
        "-E", "quote=n",
        "-E", "occurrence=f",
    ]

    if display_filter:
        cmd += ["-Y", display_filter]

    for field in tshark_fields:
        cmd += ["-e", field]

    with open(output_tsv, "w") as fh:
        subprocess.run(cmd, stdout=fh, check=True)


def _read_tshark_tsv(path: Path) -> pd.DataFrame:
    """
    Lee el TSV generado por tshark de forma rápida.
    - na_filter=False: no escanea NaNs (más rápido) y deja vacíos como ''.
    """
    try:
        df = pd.read_csv(
            path,
            sep="	",
            engine="c",
            on_bad_lines="skip",
            # Leer como texto evita inferencia (puertos como int) y elimina DtypeWarning.
            dtype=str,
            na_filter=False,
            keep_default_na=False,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if df.empty:
        return df

    # Normaliza tipos para cálculos (tiempos/longitudes)
    if "frame.time_epoch" in df.columns:
        df["frame.time_epoch"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
        df = df.dropna(subset=["frame.time_epoch"]).copy()

    if "frame.len" in df.columns:
        df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce").fillna(0).astype("int64")
    else:
        df["frame.len"] = 0

    return df


def _direction_mask(df: pd.DataFrame, protocol: str, transport: str,
                    ip_src: str, ip_dst: str,
                    src_port=None, dst_port=None) -> pd.Series:
    """
    Máscara para quedarnos sólo con paquetes de una dirección (ip/srcport -> ip/dstport).
    """
    if df.empty:
        return pd.Series([], dtype=bool)

    if "ip.src" not in df.columns or "ip.dst" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    m = (df["ip.src"] == str(ip_src)) & (df["ip.dst"] == str(ip_dst))

    pp = _port_prefix(protocol, transport)

    if src_port is not None:
        col = f"{pp}.srcport"
        if col in df.columns:
            m &= (df[col] == str(src_port))
        else:
            m &= False

    if dst_port is not None:
        col = f"{pp}.dstport"
        if col in df.columns:
            m &= (df[col] == str(dst_port))
        else:
            m &= False

    return m


def _compute_metrics_from_frames(flow, direction_label: str, df_a: pd.DataFrame, df_b: pd.DataFrame):
    """
    Calcula métricas a partir de dos DataFrames ya filtrados para una dirección:
    df_a = captura lado origen, df_b = captura lado destino.
    """
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        print(f"   ⚠️  {direction_label}: no se encontraron paquetes que cumplan el filtro.")
        return None

    key_fields = flow["key_fields"]
    jitter_enabled = flow.get("jitter", False)

    # Normaliza nombres de tiempo
    df_a = df_a.rename(columns={"frame.time_epoch": "t_a"})
    df_b = df_b.rename(columns={"frame.time_epoch": "t_b"})

    # Asegura tipos numéricos para orden/agrupación
    df_a["t_a"] = pd.to_numeric(df_a["t_a"], errors="coerce")
    df_b["t_b"] = pd.to_numeric(df_b["t_b"], errors="coerce")
    df_a = df_a.dropna(subset=["t_a"]).copy()
    df_b = df_b.dropna(subset=["t_b"]).copy()

    if df_a.empty or df_b.empty:
        print(f"   ⚠️  {direction_label}: sin timestamps válidos; se omite.")
        return None

    df_a = df_a.sort_values("t_a").reset_index(drop=True)
    df_b = df_b.sort_values("t_b").reset_index(drop=True)

    # Garantiza frame.len
    if "frame.len" not in df_a:
        df_a["frame.len"] = 0
    if "frame.len" not in df_b:
        df_b["frame.len"] = 0

    # 1) construir key (hash vectorizado)
    df_a["key"] = build_key(df_a, key_fields)
    df_b["key"] = build_key(df_b, key_fields)

    # 2) enumerar repetidos por key
    df_a["idx"] = df_a.groupby("key").cumcount()
    df_b["idx"] = df_b.groupby("key").cumcount()

    # 3) match 1-a-1 por (key, idx)
    merged = df_a[["key", "idx", "t_a"]].merge(
        df_b[["key", "idx", "t_b"]],
        on=["key", "idx"],
        how="inner",
    )

    if merged.empty:
        print(f"   ⚠️  {direction_label}: no se encontraron pares coincidentes entre ambas capturas.")
        return None

    merged["delay"] = merged["t_b"] - merged["t_a"]
    merged["t_sec"] = merged["t_a"].astype(int)

    # Filtrar retardos negativos y outliers respecto a la media
    merged = merged[merged["delay"] >= 0].copy()
    if merged.empty:
        print(f"   ⚠️  {direction_label}: todos los retardos fueron negativos; se omite.")
        return None

    delay_mean = merged["delay"].mean()
    delay_std = merged["delay"].std()
    if pd.notna(delay_std) and delay_std > 0:
        delay_upper = delay_mean + OUTLIER_STD_FACTOR * delay_std
        merged = merged[merged["delay"] <= delay_upper].copy()
        if merged.empty:
            print(f"   ⚠️  {direction_label}: todos los retardos se descartaron como outliers; se omite.")
            return None

    # Delay por segundo (media ponderada por conteo)
    delay_stats = (
        merged.groupby("t_sec")["delay"]
        .agg(["count", "mean", "min", "max", "sum"])
        .reset_index()
        .rename(columns={
            "count": "delay_count",
            "mean": "delay_mean",
            "min": "delay_min",
            "max": "delay_max",
            "sum": "delay_sum",
        })
    )

    delay_count = int(delay_stats["delay_count"].sum())
    delay_sum = float(delay_stats["delay_sum"].sum())
    delay_min = float(delay_stats["delay_min"].min()) if not delay_stats.empty else 0.0
    delay_max = float(delay_stats["delay_max"].max()) if not delay_stats.empty else 0.0

    # ========= jitter (opcional) =========
    jitter_stats = None
    jitter_count = 0
    jitter_sum = 0.0
    jitter_max = 0.0

    if jitter_enabled:
        # Calcula jitter estilo RFC3550 (estimador exponencial)
        merged_sorted = merged.sort_values("t_a").reset_index(drop=True)
        delays = merged_sorted["delay"].tolist()
        if len(delays) >= 2:
            jitters = [0.0]
            prev_j = 0.0
            prev_delay = delays[0]
            for d in delays[1:]:
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
                    merged_sorted.groupby("t_sec")["jitter"]
                    .agg(["count", "mean", "max", "sum"])
                    .reset_index()
                    .rename(columns={
                        "count": "jitter_count",
                        "mean": "jitter_mean",
                        "max": "jitter_max",
                        "sum": "jitter_sum",
                    })
                )
                jitter_count = int(jitter_stats["jitter_count"].sum())
                jitter_sum = float(jitter_stats["jitter_sum"].sum())
                jitter_max = float(jitter_stats["jitter_max"].max()) if not jitter_stats.empty else 0.0

    # ========= pérdida =========
    # Conteos por segundo en cada lado (antes y después)
    count_a = df_a.assign(t_sec=df_a["t_a"].astype(int)).groupby("t_sec").size().rename("count_a")
    count_b = df_b.assign(t_sec=df_b["t_b"].astype(int)).groupby("t_sec").size().rename("count_b")

    loss_df = (
        pd.concat([count_a, count_b], axis=1, keys=["count_a", "count_b"])
        .fillna(0)
    )
    loss_df["lost"] = loss_df["count_a"] - loss_df["count_b"]
    loss_df = loss_df.reset_index()

    # ========= throughput =========
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
            "jitter_count": "jitter_count",
            "jitter_mean": "jitter_mean",
            "jitter_max": "jitter_max",
            "jitter_sum": "jitter_sum",
        })
        result = result.merge(jitter_stats, on="t_sec", how="left")

    result = result.fillna(0)

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
        "direction": direction_label,
    }
def analyze_direction(flow, direction_info):
    """
    Analiza una dirección (A->B o B->A) y devuelve métricas y DataFrames.

    Mantiene la misma salida que antes, pero:
      - usa TSV (más rápido de parsear),
      - temporales en /dev/shm si está disponible.
    """
    protocol = flow["protocol"]
    transport = flow["transport"]
    extra_filter = flow.get("extra_filter", "")
    key_fields = flow["key_fields"]

    # Filtro direccional (tal cual versión anterior)
    display_filter = build_display_filter(
        protocol,
        transport,
        direction_info["ip_src"],
        direction_info["ip_dst"],
        direction_info.get("src_port"),
        direction_info.get("dst_port"),
        extra_filter,
    )

    # Asegura campos mínimos para poder separar/depurar
    pp = _port_prefix(protocol, transport)
    required = ["ip.src", "ip.dst", f"{pp}.srcport", f"{pp}.dstport"]
    extract_fields = list(dict.fromkeys(list(key_fields) + required))

    with tempfile.TemporaryDirectory(dir=_tmp_base_dir()) as tmpdir:
        tmp_a = Path(tmpdir) / "side_a.tsv"
        tmp_b = Path(tmpdir) / "side_b.tsv"

        run_tshark_to_tsv(direction_info["pcap_src"], display_filter, extract_fields, tmp_a)
        run_tshark_to_tsv(direction_info["pcap_dst"], display_filter, extract_fields, tmp_b)

        df_a = _read_tshark_tsv(tmp_a)
        df_b = _read_tshark_tsv(tmp_b)

    return _compute_metrics_from_frames(flow, direction_info["direction"], df_a, df_b)
def analyze_flow(flow, output_dir):
    name = flow["name"]
    print(f"\n=== Analizando flujo: {name} (bidireccional optimizado) ===")

    protocol = flow["protocol"]
    transport = flow["transport"]
    extra_filter = flow.get("extra_filter", "")
    key_fields = flow["key_fields"]

    # Construye filtro bidireccional: (A->B) || (B->A)
    fwd_base = build_display_filter(
        protocol, transport,
        flow["ip_src"], flow["ip_dst"],
        flow.get("src_port"), flow.get("dst_port"),
        extra="",  # el extra se aplica al final (una sola vez)
    )
    rev_base = build_display_filter(
        protocol, transport,
        flow["ip_dst"], flow["ip_src"],
        flow.get("dst_port"), flow.get("src_port"),
        extra="",
    )

    display_filter = f"({fwd_base}) || ({rev_base})"
    if extra_filter:
        display_filter = f"({display_filter}) && ({extra_filter})"

    # Extraer una sola vez por cada pcap
    pp = _port_prefix(protocol, transport)
    required = ["ip.src", "ip.dst", f"{pp}.srcport", f"{pp}.dstport"]
    extract_fields = list(dict.fromkeys(list(key_fields) + required))

    with tempfile.TemporaryDirectory(dir=_tmp_base_dir()) as tmpdir:
        tmp_a = Path(tmpdir) / "side_a.tsv"
        tmp_b = Path(tmpdir) / "side_b.tsv"

        run_tshark_to_tsv(flow["pcap_a"], display_filter, extract_fields, tmp_a)
        run_tshark_to_tsv(flow["pcap_b"], display_filter, extract_fields, tmp_b)

        df_side_a = _read_tshark_tsv(tmp_a)
        df_side_b = _read_tshark_tsv(tmp_b)

    if df_side_a.empty or df_side_b.empty:
        print("   ⚠️  Sin datos en uno de los PCAP (tras filtro bidireccional); se omite.")
        return

    # Separa direcciones en memoria (sin volver a correr tshark)
    src_port = flow.get("src_port")
    dst_port = flow.get("dst_port")

    mask_fwd_a = _direction_mask(df_side_a, protocol, transport, flow["ip_src"], flow["ip_dst"], src_port, dst_port)
    mask_fwd_b = _direction_mask(df_side_b, protocol, transport, flow["ip_src"], flow["ip_dst"], src_port, dst_port)

    mask_rev_b = _direction_mask(df_side_b, protocol, transport, flow["ip_dst"], flow["ip_src"], dst_port, src_port)
    mask_rev_a = _direction_mask(df_side_a, protocol, transport, flow["ip_dst"], flow["ip_src"], dst_port, src_port)

    metrics = []

    # Forward: src=pcap_a, dst=pcap_b
    fwd_metrics = _compute_metrics_from_frames(flow, "forward", df_side_a.loc[mask_fwd_a].copy(), df_side_b.loc[mask_fwd_b].copy())
    if fwd_metrics is not None:
        metrics.append(fwd_metrics)

    # Reverse: src=pcap_b, dst=pcap_a
    rev_metrics = _compute_metrics_from_frames(flow, "reverse", df_side_b.loc[mask_rev_b].copy(), df_side_a.loc[mask_rev_a].copy())
    if rev_metrics is not None:
        metrics.append(rev_metrics)

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
        row = {"t_sec": t_sec}

        # delay
        row["delay_count"] = grp["delay_count"].sum()
        row["delay_sum"] = grp["delay_sum"].sum()
        row["delay_min"] = grp["delay_min"].min() if len(grp) else 0
        row["delay_max"] = grp["delay_max"].max() if len(grp) else 0
        row["delay_mean"] = (row["delay_sum"] / row["delay_count"]) if row["delay_count"] else 0

        # jitter
        if "jitter_count" in grp.columns:
            row["jitter_count"] = grp["jitter_count"].sum()
            row["jitter_sum"] = grp["jitter_sum"].sum() if "jitter_sum" in grp else 0
            row["jitter_max"] = grp["jitter_max"].max() if "jitter_max" in grp else 0
            row["jitter_mean"] = (row["jitter_sum"] / row["jitter_count"]) if row["jitter_count"] else 0

        # loss + throughput
        row["count_a"] = grp["count_a"].sum() if "count_a" in grp else 0
        row["count_b"] = grp["count_b"].sum() if "count_b" in grp else 0
        row["lost"] = grp["lost"].sum() if "lost" in grp else 0
        row["throughput_bytes"] = grp["throughput_bytes"].sum() if "throughput_bytes" in grp else 0

        row["endpoints"] = grp["endpoints"].iloc[0]
        agg_rows.append(row)

    out_df = pd.DataFrame(agg_rows).sort_values("t_sec")
    out_df.to_csv(output_file, index=False)

    # Resumen
    total_lost = out_df["lost"].sum() if "lost" in out_df else 0
    total_lost = max(total_lost, 0)
    total_throughput = out_df["throughput_bytes"].sum() if "throughput_bytes" in out_df else 0

    total_delay_count = out_df["delay_count"].sum() if "delay_count" in out_df else 0
    total_delay_sum = out_df["delay_sum"].sum() if "delay_sum" in out_df else 0
    total_delay_mean = (total_delay_sum / total_delay_count) if total_delay_count else 0

    total_jitter_count = out_df["jitter_count"].sum() if "jitter_count" in out_df else 0
    total_jitter_sum = out_df["jitter_sum"].sum() if "jitter_sum" in out_df else 0
    total_jitter_mean = (total_jitter_sum / total_jitter_count) if total_jitter_count else 0
    total_jitter_max = out_df["jitter_max"].max() if "jitter_max" in out_df else 0

    print(f"   ✅ Guardado: {output_file}")
    print(f"   📈 Delay mean (s): {total_delay_mean}")
    if flow.get("jitter", False):
        print(f"   📉 Jitter mean (s): {total_jitter_mean} | max (s): {total_jitter_max}")
    print(f"   📦 Lost (pkts bidireccionales): {total_lost}")
    print(f"   🚚 Throughput B (bytes totales bidireccionales): {total_throughput}")
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
