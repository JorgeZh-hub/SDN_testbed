#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_csv_maybe_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def natural_load_key(s: str):
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", str(s))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return s

def read_flow_csv(path: Path) -> pd.DataFrame:
    """
    Algunos CSV pueden venir con coma o tab. Autodetect simple:
    - si read_csv() produce 1 columna, intenta sep='\\t'
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")
    return df

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def t_critical(level: float, df: int) -> float:
    alpha = 1.0 - level
    p = 1.0 - alpha / 2.0
    try:
        from scipy.stats import t
        return float(t.ppf(p, df))
    except Exception:
        # fallback común para 95%
        return 1.96

def compute_ci(mean: float, std: float, n: int, level: float, method: str) -> Tuple[Optional[float], Optional[float]]:
    if n < 2 or mean is None or std is None or pd.isna(mean) or pd.isna(std):
        return (None, None)
    se = std / math.sqrt(n)
    crit = t_critical(level, n - 1) if method == "t" else 1.96
    return (mean - crit * se, mean + crit * se)

def unit_throughput_factor(units: str) -> float:
    # bytes/s -> Mbps or Kbps
    if units.lower() == "mbps":
        return 8.0 / 1e6
    if units.lower() == "kbps":
        return 8.0 / 1e3
    return 8.0 / 1e6  # default Mbps

def slugify(val: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(val))
    s = s.strip("_")
    return s or "all"

def maybe_smooth(series: pd.Series, smooth_cfg: dict) -> pd.Series:
    if not smooth_cfg or not smooth_cfg.get("enable", False):
        return series
    try:
        w = int(smooth_cfg.get("window_points", 3))
    except Exception:
        w = 3
    if w <= 1:
        return series
    return series.rolling(window=w, min_periods=1, center=True).mean()


# -------------------------
# Warmup auto (por count)
# -------------------------

def detect_warmup_count(df_1s: pd.DataFrame, cfg_auto: dict) -> Optional[int]:
    th = int(cfg_auto.get("count_th", 1))
    sustain = int(cfg_auto.get("sustain_s", 3))
    max_search = int(cfg_auto.get("max_search_s", 120))

    if "count" not in df_1s.columns or "t_rel_s" not in df_1s.columns:
        return None

    d = df_1s[df_1s["t_rel_s"] <= max_search].copy()
    if d.empty:
        return None

    c = to_num(d["count"]).fillna(0).astype(int)
    active = (c >= th).astype(int)
    run = active.rolling(window=sustain, min_periods=sustain).sum()

    hits = d.loc[run >= sustain, "t_rel_s"]
    if hits.empty:
        return None

    first_hit = int(hits.iloc[0])
    warmup = max(0, first_hit - (sustain - 1))
    return warmup


# -------------------------
# Core compute per file/rep
# -------------------------

def compute_file_rep_summaries(
    df_raw: pd.DataFrame,
    cfg: dict,
    flow_id_fallback: str,
) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - flow_summary (post-filter)
      - proto_accum (para sumar a protocolo en este rep)
      - df_ts_flow (t_rel_s, throughput_Mbps) para throughput vs tiempo del flow
      - df_ts_rows_proto (t_sec, protocol, throughput_bytes) para agregación por protocolo en rep
    """
    filters = cfg["filters"]
    warm = cfg["warmup"]
    loss_cfg = cfg["loss"]
    thr_cfg = cfg["throughput"]
    units = cfg["units"]

    thr_factor = unit_throughput_factor(units.get("throughput", "Mbps"))

    # columnas típicas
    for col in ["flow", "protocol", "transport", "endpoints", "t_sec", "count", "mean",
                "count_a", "count_b", "lost", "throughput_bytes",
                "jitter_count", "jitter_mean"]:
        if col not in df_raw.columns:
            df_raw[col] = pd.NA

    df = df_raw.copy()
    df["t_sec"] = to_num(df["t_sec"])
    df = df.dropna(subset=["t_sec"]).sort_values("t_sec")
    if df.empty:
        return {}, {}, pd.DataFrame(), pd.DataFrame()

    # IDs
    flow_name = df["flow"].dropna().astype(str).iloc[0] if df["flow"].notna().any() else flow_id_fallback
    protocol = df["protocol"].dropna().astype(str).iloc[0] if df["protocol"].notna().any() else "unknown"

    # t_rel
    t0 = float(df["t_sec"].iloc[0])
    df["t_rel_s"] = (df["t_sec"] - t0).astype(int)

    # Warmup per flow
    warmup_s = 0
    if warm.get("mode", "none") == "fixed":
        warmup_s = int(warm.get("fixed_seconds", 0))
    elif warm.get("mode", "none") == "auto":
        w = detect_warmup_count(df[["t_rel_s", "count"]].copy(), warm.get("auto", {}))
        warmup_s = int(warm.get("fallback_if_not_found", 0)) if w is None else int(w)
    else:
        warmup_s = 0

    # filtros base para métricas delay/jitter
    min_count = int(filters.get("min_count_per_row", 1))
    df["count"] = to_num(df["count"]).fillna(0)
    df["mean"] = to_num(df["mean"])
    df["throughput_bytes"] = to_num(df["throughput_bytes"]).fillna(0)

    df_used = df[df["t_rel_s"] >= warmup_s].copy()
    df_used = df_used[df_used["count"] >= min_count].copy()
    if filters.get("drop_rows_with_nan_mean", True):
        df_used = df_used[df_used["mean"].notna()].copy()

    # pérdida
    sent_basis = loss_cfg.get("sent_basis", "count_a")
    if sent_basis not in df.columns:
        sent_basis = "count"

    df_used["sent"] = to_num(df_used.get(sent_basis, pd.Series(dtype=float))).fillna(df_used["count"])
    df_used["lost"] = to_num(df_used.get("lost", pd.Series(dtype=float))).fillna(0)

    include_neg = bool(loss_cfg.get("include_negative_loss", False))
    if include_neg:
        lost_adj = df_used["lost"]
    else:
        lost_adj = df_used["lost"].clip(lower=0)

    # jitter (puede no existir)
    has_jitter = df_used["jitter_mean"].notna().any() and df_used["jitter_count"].notna().any()
    df_used["jitter_mean"] = to_num(df_used["jitter_mean"])
    df_used["jitter_count"] = to_num(df_used["jitter_count"]).fillna(0)

    # ---- Delay global ponderado por count ----
    delay_count = df_used["count"].sum()
    delay_weight = (df_used["mean"] * df_used["count"]).sum() if delay_count > 0 else None
    delay_mean_s = (delay_weight / delay_count) if (delay_count and delay_weight is not None) else None
    delay_mean_ms = (delay_mean_s * 1000.0) if delay_mean_s is not None else None

    # ---- Jitter global ponderado por jitter_count ----
    jitter_mean_ms = None
    jitter_count_total = 0
    if has_jitter:
        jitter_count_total = df_used["jitter_count"].sum()
        jw = (df_used["jitter_mean"] * df_used["jitter_count"]).sum() if jitter_count_total > 0 else None
        jm_s = (jw / jitter_count_total) if (jitter_count_total and jw is not None) else None
        jitter_mean_ms = (jm_s * 1000.0) if jm_s is not None else None

    # ---- Throughput promedio ----
    # total bytes post-warmup (ojo: throughput_bytes ya es "por segundo")
    total_bytes = df_used["throughput_bytes"].sum()

    if thr_cfg.get("normalization", "wall") == "active":
        denom_s = max(1, int(df_used.shape[0]))
    else:
        # wall duration en segundos (t_max - t_min + 1)
        denom_s = int(df_used["t_rel_s"].max() - df_used["t_rel_s"].min() + 1) if df_used.shape[0] >= 2 else 1

    throughput_mean = (total_bytes / denom_s) * thr_factor  # Mbps o Kbps según factor
    thr_unit = cfg["units"].get("throughput", "Mbps")

    # ---- Loss % ----
    sent_total = df_used["sent"].sum()
    lost_total = float(lost_adj.sum()) if not df_used.empty else 0.0
    loss_pct = (100.0 * lost_total / sent_total) if sent_total > 0 else None

    flow_summary = {
        "id": flow_name,
        "protocol": protocol,
        "warmup_s": warmup_s,
        "delay_mean_ms": delay_mean_ms,
        "delay_weight_count": float(delay_count),

        "jitter_mean_ms": jitter_mean_ms,
        "jitter_weight_count": float(jitter_count_total) if has_jitter else 0.0,

        f"throughput_total_mean_{thr_unit}": throughput_mean,
        "throughput_total_bytes": float(total_bytes),
        "throughput_denom_s": float(denom_s),

        "lost_total_pkts": float(lost_total),
        "sent_total_pkts": float(sent_total),
        "loss_pct": loss_pct,
        "rows_used": int(df_used.shape[0]),
    }

    # Para agregación por protocolo en el rep
    proto_accum = {
        "protocol": protocol,
        "delay_weight_sum_s": float((df_used["mean"] * df_used["count"]).sum()) if delay_count > 0 else 0.0,
        "delay_count_sum": float(delay_count),
        "jitter_weight_sum_s": float((df_used["jitter_mean"] * df_used["jitter_count"]).sum()) if has_jitter else 0.0,
        "jitter_count_sum": float(jitter_count_total) if has_jitter else 0.0,
        "throughput_total_bytes": float(total_bytes),
        "t_sec_min": float(df_used["t_sec"].min()) if not df_used.empty else None,
        "t_sec_max": float(df_used["t_sec"].max()) if not df_used.empty else None,
        "lost_total_pkts": float(lost_total),
        "sent_total_pkts": float(sent_total),
    }

    # Timeseries por flow (throughput vs tiempo)
    df_ts = df.copy() if cfg["timeseries"].get("include_warmup", True) else df[df["t_rel_s"] >= warmup_s].copy()
    df_ts = df_ts[["t_rel_s", "throughput_bytes"]].copy()
    df_ts["throughput_Mbps"] = df_ts["throughput_bytes"] * unit_throughput_factor("Mbps")  # para serie fija en Mbps
    df_ts = df_ts.drop(columns=["throughput_bytes"])

    # filas para construir timeseries por protocolo (sumar bytes por segundo)
    df_ts_proto_rows = df.copy() if cfg["timeseries"].get("include_warmup", True) else df[df["t_rel_s"] >= warmup_s].copy()
    df_ts_proto_rows = df_ts_proto_rows[["t_sec", "protocol", "throughput_bytes"]].copy()

    return flow_summary, proto_accum, df_ts, df_ts_proto_rows


# -------------------------
# Plot helpers
# -------------------------

def plot_lines_vs_load(df: pd.DataFrame, out_path: Path, metric_base: str, title: str):
    """
    df debe tener (por id ya filtrado):
      controller, load,
      {metric_base}_mean, {metric_base}_ci95_low, {metric_base}_ci95_high
    """
    mean_col = f"{metric_base}_mean"
    low_col  = f"{metric_base}_ci95_low"
    high_col = f"{metric_base}_ci95_high"

    fig, ax = plt.subplots()

    loads = sorted(df["load"].dropna().unique().tolist(), key=natural_load_key)
    x = list(range(len(loads)))
    ax.set_xticks(x)
    ax.set_xticklabels(loads)

    controllers = sorted(df["controller"].dropna().unique().tolist())
    for ctrl in controllers:
        d = df[df["controller"] == ctrl].copy()
        # alinea a loads
        d["__k"] = d["load"].apply(natural_load_key)
        d = d.sort_values("__k")

        # crea arrays alineados
        y = []
        lo = []
        hi = []
        for ld in loads:
            row = d[d["load"] == ld]
            if row.empty:
                y.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))
            else:
                y.append(row.iloc[0].get(mean_col, float("nan")))
                lo.append(row.iloc[0].get(low_col, float("nan")))
                hi.append(row.iloc[0].get(high_col, float("nan")))

        ax.plot(x, y, marker="o", label=ctrl)

        # sombra CI (solo donde exista)
        lo_s = pd.to_numeric(pd.Series(lo), errors="coerce")
        hi_s = pd.to_numeric(pd.Series(hi), errors="coerce")
        y_s  = pd.to_numeric(pd.Series(y),  errors="coerce")
        mask = lo_s.notna() & hi_s.notna() & y_s.notna()
        if mask.any():
            ax.fill_between(pd.Series(x)[mask], lo_s[mask], hi_s[mask], alpha=0.15)

    ax.set_xlabel("load")
    ax.set_ylabel(metric_base)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_grouped_bars(df: pd.DataFrame, out_path: Path, metric_base: str, title: str):
    """
    df debe tener:
      controller, load,
      {metric_base}_mean, {metric_base}_ci95_low, {metric_base}_ci95_high
    """
    mean_col = f"{metric_base}_mean"
    low_col  = f"{metric_base}_ci95_low"
    high_col = f"{metric_base}_ci95_high"

    loads = sorted(df["load"].dropna().unique().tolist(), key=natural_load_key)
    controllers = sorted(df["controller"].dropna().unique().tolist())

    width = 0.8
    n_ctrl = max(1, len(controllers))
    bar_w = width / n_ctrl
    x = list(range(len(loads)))

    fig, ax = plt.subplots()

    for j, ctrl in enumerate(controllers):
        ys = []
        yerr_low = []
        yerr_high = []

        for ld in loads:
            row = df[(df["controller"] == ctrl) & (df["load"] == ld)]
            if row.empty:
                ys.append(float("nan"))
                yerr_low.append(0.0); yerr_high.append(0.0)
                continue

            m = row.iloc[0].get(mean_col, float("nan"))
            lo = row.iloc[0].get(low_col, float("nan"))
            hi = row.iloc[0].get(high_col, float("nan"))

            ys.append(m)

            if pd.notna(m) and pd.notna(lo) and pd.notna(hi):
                yerr_low.append(max(0.0, m - lo))
                yerr_high.append(max(0.0, hi - m))
            else:
                yerr_low.append(0.0); yerr_high.append(0.0)

        offsets = [xi - width/2 + (j + 0.5)*bar_w for xi in x]
        ax.bar(offsets, ys, bar_w, yerr=[yerr_low, yerr_high], capsize=3, label=ctrl)

    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.set_xlabel("load")
    ax.set_ylabel(metric_base)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_multi(ts_df: pd.DataFrame, out_path: Path, metric: str, min_reps_per_t: int,
                          crop_to_common_time: bool = False):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"
    count_col = f"{metric}_count"

    df = ts_df.copy().sort_values(["controller", "t_rel_s"])
    if count_col in df.columns:
        df = df[df[count_col] >= min_reps_per_t]
    df = df.dropna(subset=[mean_col])
    if df.empty:
        print(f"WARNING: serie vacía -> {out_path}")
        return

    # ---- RECORTE AL TIEMPO COMÚN ----
    if crop_to_common_time:
        tmax_by_ctrl = df.groupby("controller")["t_rel_s"].max().dropna()
        if not tmax_by_ctrl.empty:
            common_tmax = int(tmax_by_ctrl.min())
            df = df[df["t_rel_s"] <= common_tmax]

    fig, ax = plt.subplots()

    for ctrl in sorted(df["controller"].dropna().unique().tolist()):
        d = df[df["controller"] == ctrl].sort_values("t_rel_s")
        if d.empty:
            continue
        ax.plot(d["t_rel_s"], d[mean_col], label=ctrl)

        # sombra CI
        if low_col in d.columns and high_col in d.columns:
            dd = d.dropna(subset=[low_col, high_col, mean_col])
            if not dd.empty:
                ax.fill_between(dd["t_rel_s"], dd[low_col], dd[high_col], alpha=0.15)

    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs tiempo (promedio sobre reps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_compare_ids(ts_df: pd.DataFrame, out_path: Path, ids: List[str], controllers: List[str],
                                id_labels: dict, labels_cfg: dict, show_ci: bool, min_reps_per_t: int,
                                crop_common_time: bool, smooth_cfg: dict):
    mean_col = "throughput_Mbps_mean"
    low_col = "throughput_Mbps_ci95_low"
    high_col = "throughput_Mbps_ci95_high"
    count_col = "throughput_Mbps_count"

    df = ts_df.copy().sort_values(["controller", "id", "t_rel_s"])
    if count_col in df.columns:
        df = df[df[count_col] >= min_reps_per_t]
    df = df.dropna(subset=[mean_col])
    if df.empty:
        print(f"WARNING: serie vacía -> {out_path}")
        return

    series = []
    for ctrl in controllers:
        for idv in ids:
            d = df[(df["controller"] == ctrl) & (df["id"] == idv)].sort_values("t_rel_s")
            if not d.empty:
                series.append((ctrl, idv, d))

    if not series:
        print(f"WARNING: no hay datos para ids={ids} controllers={controllers} -> {out_path}")
        return

    if crop_common_time:
        tmax_list = [int(s[2]["t_rel_s"].max()) for s in series if not s[2].empty]
        if tmax_list:
            common_tmax = min(tmax_list)
            series = [(c, i, d[d["t_rel_s"] <= common_tmax].copy()) for (c, i, d) in series]

    fig, ax = plt.subplots()
    multi_ctrl = len({c for c, _, _ in series}) > 1

    for ctrl, idv, d in series:
        label_id = id_labels.get(idv, idv)
        label = f"{label_id} ({ctrl})" if multi_ctrl else label_id

        mean_s = maybe_smooth(d[mean_col], smooth_cfg)
        ax.plot(d["t_rel_s"], mean_s, label=label)

        if show_ci and low_col in d.columns and high_col in d.columns:
            low_s = maybe_smooth(d[low_col], smooth_cfg)
            high_s = maybe_smooth(d[high_col], smooth_cfg)
            mask = low_s.notna() & high_s.notna()
            if mask.any():
                ax.fill_between(d.loc[mask, "t_rel_s"], low_s[mask], high_s[mask], alpha=0.15)

    title_tpl = (labels_cfg or {}).get("title", "Throughput vs tiempo")
    ctrl_title = controllers[0] if len(controllers) == 1 else "all"
    try:
        title = title_tpl.format(id=", ".join(ids), controller=ctrl_title)
    except Exception:
        title = title_tpl

    ax.set_title(title)
    ax.set_xlabel((labels_cfg or {}).get("xlabel", "Tiempo (s)"))
    ax.set_ylabel((labels_cfg or {}).get("ylabel", "Throughput (Mbps)"))
    legend_title = (labels_cfg or {}).get("legend", "")
    if legend_title:
        ax.legend(title=legend_title)
    else:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reports_flows.yml")
    ap.add_argument("--regenerate-csvs", action="store_true", help="Recalcula y reescribe CSVs aunque ya existan.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    extracted_root = Path(cfg["inputs"]["extracted_root"]).resolve()
    flows_glob = cfg["inputs"]["flows_glob"]
    flow_filter = cfg["inputs"].get("flow_filter", []) or []
    group_by = cfg["inputs"].get("group_by", ["flow"])
    thr_unit = cfg["units"].get("throughput", "Mbps")

    out_dir = Path(cfg["outputs"]["out_dir"]).resolve()
    plots_dir = Path(cfg["outputs"]["plots_dir"]).resolve()
    safe_mkdir(out_dir)
    safe_mkdir(plots_dir)

    # outputs
    p_rep_flow = out_dir / cfg["outputs"]["csv"]["rep_summary_flow"]
    p_rep_proto = out_dir / cfg["outputs"]["csv"]["rep_summary_protocol"]
    p_load_flow = out_dir / cfg["outputs"]["csv"]["load_summary_flow"]
    p_load_proto = out_dir / cfg["outputs"]["csv"]["load_summary_protocol"]
    p_ts_flow = out_dir / cfg["outputs"]["csv"]["ts_throughput_flow"]
    p_ts_proto = out_dir / cfg["outputs"]["csv"]["ts_throughput_protocol"]
    p_missing = out_dir / cfg["outputs"]["csv"]["missing"]

    required_csvs = [
        p_rep_flow, p_rep_proto, p_load_flow, p_load_proto,
        p_ts_flow, p_ts_proto,
    ]
    use_existing_csvs = (not args.regenerate_csvs and all(p.exists() for p in required_csvs))
    if use_existing_csvs:
        print("INFO: usando CSVs existentes (usa --regenerate-csvs para regenerarlos).")
        rep_flow_df = read_csv_maybe_empty(p_rep_flow)
        rep_proto_df = read_csv_maybe_empty(p_rep_proto)
        load_flow_df = read_csv_maybe_empty(p_load_flow)
        load_proto_df = read_csv_maybe_empty(p_load_proto)
        ts_flow_mean_df = read_csv_maybe_empty(p_ts_flow)
        ts_proto_mean_df = read_csv_maybe_empty(p_ts_proto)
        miss_df = read_csv_maybe_empty(p_missing)
    else:
        missing = []
        rep_flow_rows = []
        rep_proto_acc = {}  # key=(controller,load,rep,protocol)->acc dict
    
        # timeseries collectors (por rep)
        ts_flow_rows = []   # controller, load, rep, id(flow), t_rel_s, throughput_Mbps
        ts_proto_rows_raw = []  # controller, load, rep, protocol, t_sec, throughput_bytes
    
        if not extracted_root.exists():
            print(f"ERROR: extracted_root no existe: {extracted_root}")
            return
    
        controllers = sorted([p for p in extracted_root.iterdir() if p.is_dir()])
        for ctrl_dir in controllers:
            controller = ctrl_dir.name
            loads = sorted([p for p in ctrl_dir.iterdir() if p.is_dir()], key=lambda p: natural_load_key(p.name))
    
            for load_dir in loads:
                load = load_dir.name
                csvs = list(load_dir.glob(flows_glob))
                if not csvs:
                    missing.append({"controller": controller, "load": load, "rep": None, "file": None, "reason": "no_flow_csvs"})
                    continue
    
                for csv_path in sorted(csvs):
                    rep_m = re.search(r"(rep_\d+)", str(csv_path))
                    rep = rep_m.group(1) if rep_m else "rep_unknown"
                    flow_id = csv_path.stem
    
                    if flow_filter and flow_id not in flow_filter:
                        continue
    
                    try:
                        df_raw = read_flow_csv(csv_path)
                    except Exception as e:
                        missing.append({"controller": controller, "load": load, "rep": rep, "file": str(csv_path), "reason": f"read_error:{e}"})
                        continue
    
                    if df_raw.empty:
                        missing.append({"controller": controller, "load": load, "rep": rep, "file": str(csv_path), "reason": "empty_csv"})
                        continue
    
                    flow_sum, proto_accum, df_ts_flow, df_ts_proto_rows = compute_file_rep_summaries(df_raw, cfg, flow_id_fallback=flow_id)
                    if not flow_sum:
                        missing.append({"controller": controller, "load": load, "rep": rep, "file": str(csv_path), "reason": "no_valid_rows"})
                        continue
    
                    rep_flow_rows.append({
                        "controller": controller,
                        "load": load,
                        "rep": rep,
                        **flow_sum,
                        "src_csv": str(csv_path),
                    })
    
                    # acumula protocolo (para resumen por protocolo en este rep)
                    if "protocol" in group_by:
                        key = (controller, load, rep, proto_accum["protocol"])
                        cur = rep_proto_acc.get(key)
                        if cur is None:
                            rep_proto_acc[key] = dict(proto_accum)
                        else:
                            cur["delay_weight_sum_s"] += proto_accum["delay_weight_sum_s"]
                            cur["delay_count_sum"] += proto_accum["delay_count_sum"]
                            cur["jitter_weight_sum_s"] += proto_accum["jitter_weight_sum_s"]
                            cur["jitter_count_sum"] += proto_accum["jitter_count_sum"]
                            cur["throughput_total_bytes"] += proto_accum["throughput_total_bytes"]
                            cur["lost_total_pkts"] += proto_accum["lost_total_pkts"]
                            cur["sent_total_pkts"] += proto_accum["sent_total_pkts"]
    
                            # min/max t_sec
                            for k in ["t_sec_min", "t_sec_max"]:
                                if proto_accum[k] is None:
                                    continue
                                if cur[k] is None:
                                    cur[k] = proto_accum[k]
                                else:
                                    cur[k] = min(cur[k], proto_accum[k]) if k.endswith("min") else max(cur[k], proto_accum[k])
    
                    # timeseries throughput flow
                    if "flow" in group_by and not df_ts_flow.empty:
                        # reindex missing seconds
                        fill = cfg["timeseries"].get("fill_missing_seconds", "zero")
                        tmax = int(df_ts_flow["t_rel_s"].max())
                        grid = pd.DataFrame({"t_rel_s": range(0, tmax + 1)})
                        m = grid.merge(df_ts_flow, on="t_rel_s", how="left")
                        if fill == "zero":
                            m["throughput_Mbps"] = m["throughput_Mbps"].fillna(0.0)
                        # si nan, se queda NaN
                        for _, r in m.iterrows():
                            ts_flow_rows.append({
                                "controller": controller,
                                "load": load,
                                "rep": rep,
                                "id": flow_sum["id"],
                                "t_rel_s": int(r["t_rel_s"]),
                                "throughput_Mbps": r["throughput_Mbps"],
                            })
    
                    # timeseries raw proto rows (para luego sumar por segundo)
                    if "protocol" in group_by and not df_ts_proto_rows.empty:
                        df_ts_proto_rows["controller"] = controller
                        df_ts_proto_rows["load"] = load
                        df_ts_proto_rows["rep"] = rep
                        ts_proto_rows_raw.append(df_ts_proto_rows)
    
        # ---- DataFrames ----
        rep_flow_df = pd.DataFrame(rep_flow_rows)
        miss_df = pd.DataFrame(missing)
    
        # protocolo rep summary
        rep_proto_rows = []
        if rep_proto_acc:
            thr_unit = cfg["units"].get("throughput", "Mbps")
            thr_factor = unit_throughput_factor(thr_unit)
    
            for (controller, load, rep, proto), a in rep_proto_acc.items():
                delay_ms = (a["delay_weight_sum_s"] / a["delay_count_sum"] * 1000.0) if a["delay_count_sum"] > 0 else None
                jitter_ms = (a["jitter_weight_sum_s"] / a["jitter_count_sum"] * 1000.0) if a["jitter_count_sum"] > 0 else None
    
                # throughput promedio por protocolo (bytes totales / wall_duration global)
                if a["t_sec_min"] is not None and a["t_sec_max"] is not None:
                    wall_s = max(1, int(a["t_sec_max"] - a["t_sec_min"] + 1))
                else:
                    wall_s = 1
                thr_mean = (a["throughput_total_bytes"] / wall_s) * thr_factor
    
                loss_pct = (100.0 * a["lost_total_pkts"] / a["sent_total_pkts"]) if a["sent_total_pkts"] > 0 else None
    
                rep_proto_rows.append({
                    "controller": controller,
                    "load": load,
                    "rep": rep,
                    "id": proto,
                    "protocol": proto,
                    "delay_mean_ms": delay_ms,
                    "jitter_mean_ms": jitter_ms,
                    f"throughput_total_mean_{thr_unit}": thr_mean,
                    "throughput_total_bytes": a["throughput_total_bytes"],
                    "lost_total_pkts": a["lost_total_pkts"],
                    "sent_total_pkts": a["sent_total_pkts"],
                    "loss_pct": loss_pct,
                    "wall_s": wall_s,
                })
    
        rep_proto_df = pd.DataFrame(rep_proto_rows)
    
        # ---- Agregación por carga (sobre reps) ----
        conf = cfg.get("stats", {}).get("confidence", {})
        ci_enable = bool(conf.get("enable", True))
        ci_level = float(conf.get("level", 0.95))
        ci_method = conf.get("method", "t")
    
        def agg_load(df_rep: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
            out = []
            if df_rep.empty:
                return pd.DataFrame()
    
            grp = df_rep.groupby(["controller", "load", "id"], dropna=False)
            for (controller, load, idv), g in grp:
                row = {"controller": controller, "load": load, "id": idv}
                for mc in metric_cols:
                    v = pd.to_numeric(g[mc], errors="coerce").dropna()
                    n = int(v.shape[0])
                    mean = float(v.mean()) if n else None
                    std = float(v.std(ddof=1)) if n >= 2 else None
                    row[f"{mc}_mean"] = mean
                    row[f"{mc}_std"] = std
                    row[f"{mc}_count"] = n
                    if ci_enable and n >= 2 and mean is not None and std is not None:
                        lo, hi = compute_ci(mean, std, n, ci_level, ci_method)
                        row[f"{mc}_ci95_low"] = lo
                        row[f"{mc}_ci95_high"] = hi
                    else:
                        row[f"{mc}_ci95_low"] = None
                        row[f"{mc}_ci95_high"] = None
                out.append(row)
    
            df_out = pd.DataFrame(out)
            df_out["__k"] = df_out["load"].apply(natural_load_key)
            df_out = df_out.sort_values(["id", "controller", "__k"]).drop(columns=["__k"])
            return df_out
    
        # métricas estándar en rep_flow
        thr_unit = cfg["units"].get("throughput", "Mbps")
        thr_col = f"throughput_total_mean_{thr_unit}"
    
    
        flow_metric_cols = ["delay_mean_ms", "jitter_mean_ms", thr_col, "loss_pct"]
        proto_metric_cols = ["delay_mean_ms", "jitter_mean_ms", thr_col, "loss_pct"]
    
        load_flow_df = agg_load(rep_flow_df, flow_metric_cols) if not rep_flow_df.empty else pd.DataFrame()
        load_proto_df = agg_load(rep_proto_df, proto_metric_cols) if not rep_proto_df.empty else pd.DataFrame()
    
        # ---- Timeseries throughput (flow) promedio sobre reps ----
        ts_flow_rep_df = pd.DataFrame(ts_flow_rows)
        ts_flow_out_rows = []
        if not ts_flow_rep_df.empty:
            g = ts_flow_rep_df.groupby(["controller", "load", "id", "t_rel_s"], dropna=False)
            for (controller, load, idv, t), gg in g:
                v = pd.to_numeric(gg["throughput_Mbps"], errors="coerce").dropna()
                n = int(v.shape[0])
                mean = float(v.mean()) if n else None
                std = float(v.std(ddof=1)) if n >= 2 else None
                row = {"controller": controller, "load": load, "id": idv, "t_rel_s": int(t)}
                row["throughput_Mbps_mean"] = mean
                row["throughput_Mbps_std"] = std
                row["throughput_Mbps_count"] = n
                if ci_enable and n >= 2 and mean is not None and std is not None:
                    lo, hi = compute_ci(mean, std, n, ci_level, ci_method)
                    row["throughput_Mbps_ci95_low"] = lo
                    row["throughput_Mbps_ci95_high"] = hi
                else:
                    row["throughput_Mbps_ci95_low"] = None
                    row["throughput_Mbps_ci95_high"] = None
                ts_flow_out_rows.append(row)
    
        ts_flow_mean_df = pd.DataFrame(ts_flow_out_rows)
        if not ts_flow_mean_df.empty:
            ts_flow_mean_df["__k"] = ts_flow_mean_df["load"].apply(natural_load_key)
            ts_flow_mean_df = ts_flow_mean_df.sort_values(["id", "controller", "__k", "t_rel_s"]).drop(columns=["__k"])
    
        # ---- Timeseries throughput (protocol) ----
        ts_proto_mean_df = pd.DataFrame()
        if "protocol" in group_by and ts_proto_rows_raw:
            raw = pd.concat(ts_proto_rows_raw, ignore_index=True)
            raw["t_sec"] = to_num(raw["t_sec"])
            raw["throughput_bytes"] = to_num(raw["throughput_bytes"]).fillna(0)
            # suma por segundo por protocolo en cada rep
            rows = []
            for (controller, load, rep, proto), gg in raw.groupby(["controller", "load", "rep", "protocol"], dropna=False):
                gg = gg.dropna(subset=["t_sec"]).sort_values("t_sec")
                if gg.empty:
                    continue
                t0 = float(gg["t_sec"].iloc[0])
                gg["t_rel_s"] = (gg["t_sec"] - t0).astype(int)
                per_s = gg.groupby("t_rel_s", as_index=False)["throughput_bytes"].sum()
                per_s["throughput_Mbps"] = per_s["throughput_bytes"] * unit_throughput_factor("Mbps")
                tmax = int(per_s["t_rel_s"].max())
                grid = pd.DataFrame({"t_rel_s": range(0, tmax + 1)})
                m = grid.merge(per_s[["t_rel_s", "throughput_Mbps"]], on="t_rel_s", how="left")
                if cfg["timeseries"].get("fill_missing_seconds", "zero") == "zero":
                    m["throughput_Mbps"] = m["throughput_Mbps"].fillna(0.0)
                for _, r in m.iterrows():
                    rows.append({
                        "controller": controller, "load": load, "rep": rep, "id": str(proto),
                        "t_rel_s": int(r["t_rel_s"]), "throughput_Mbps": r["throughput_Mbps"]
                    })
    
            rep_ts = pd.DataFrame(rows)
            out = []
            if not rep_ts.empty:
                g = rep_ts.groupby(["controller", "load", "id", "t_rel_s"], dropna=False)
                for (controller, load, idv, t), gg in g:
                    v = pd.to_numeric(gg["throughput_Mbps"], errors="coerce").dropna()
                    n = int(v.shape[0])
                    mean = float(v.mean()) if n else None
                    std = float(v.std(ddof=1)) if n >= 2 else None
                    row = {"controller": controller, "load": load, "id": idv, "t_rel_s": int(t)}
                    row["throughput_Mbps_mean"] = mean
                    row["throughput_Mbps_std"] = std
                    row["throughput_Mbps_count"] = n
                    if ci_enable and n >= 2 and mean is not None and std is not None:
                        lo, hi = compute_ci(mean, std, n, ci_level, ci_method)
                        row["throughput_Mbps_ci95_low"] = lo
                        row["throughput_Mbps_ci95_high"] = hi
                    else:
                        row["throughput_Mbps_ci95_low"] = None
                        row["throughput_Mbps_ci95_high"] = None
                    out.append(row)
    
            ts_proto_mean_df = pd.DataFrame(out)
            if not ts_proto_mean_df.empty:
                ts_proto_mean_df["__k"] = ts_proto_mean_df["load"].apply(natural_load_key)
                ts_proto_mean_df = ts_proto_mean_df.sort_values(["id", "controller", "__k", "t_rel_s"]).drop(columns=["__k"])
    
        # ---- Save CSVs ----
        rep_flow_df.to_csv(p_rep_flow, index=False)
        rep_proto_df.to_csv(p_rep_proto, index=False)
        load_flow_df.to_csv(p_load_flow, index=False)
        load_proto_df.to_csv(p_load_proto, index=False)
        ts_flow_mean_df.to_csv(p_ts_flow, index=False)
        ts_proto_mean_df.to_csv(p_ts_proto, index=False)
        miss_df.to_csv(p_missing, index=False)
    
        print(f"OK: {p_rep_flow}")
        print(f"OK: {p_rep_proto}")
        print(f"OK: {p_load_flow}")
        print(f"OK: {p_load_proto}")
        print(f"OK: {p_ts_flow}")
        print(f"OK: {p_ts_proto}")
        print(f"OK: {p_missing}")
    
    # ---- Plots ----
    plots = cfg.get("plots", {})

    def pick_load_df(group: str):
        return load_flow_df if group == "flow" else load_proto_df

    def pick_ts_df(group: str):
        return ts_flow_mean_df if group == "flow" else ts_proto_mean_df

    # 1) delay vs load
    if plots.get("delay_vs_load", {}).get("enable", False):
        pcfg = plots["delay_vs_load"]
        group = pcfg.get("group", "flow")
        metric = pcfg.get("metric", "delay_mean_ms")
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = sorted(dfL["id"].dropna().unique().tolist())
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                ycol = f"{metric}_mean"
                if ycol not in d.columns:
                    continue
                outp = plots_dir / f"delay_vs_load_{group}_{idv}.png"
                plot_lines_vs_load(d, outp, metric_base=metric, title=f"Delay vs load ({group}={idv})")
                print(f"OK plot: {outp}")

    # 2) jitter vs load (solo si existe)
    if plots.get("jitter_vs_load", {}).get("enable", False):
        pcfg = plots["jitter_vs_load"]
        group = pcfg.get("group", "flow")
        metric = pcfg.get("metric", "jitter_mean_ms")
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = sorted(dfL["id"].dropna().unique().tolist())
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                ycol = f"{metric}_mean"
                if ycol not in d.columns:
                    continue
                # si todo NaN, no plot
                if cfg.get("jitter", {}).get("plot_only_if_present", True):
                    if d[ycol].dropna().empty:
                        continue

                outp = plots_dir / f"jitter_vs_load_{group}_{idv}.png"
                plot_lines_vs_load(d, outp, metric_base=metric, title=f"Jitter vs load ({group}={idv})")
                print(f"OK plot: {outp}")

    # 3) loss bars
    if plots.get("loss_bars", {}).get("enable", False):
        pcfg = plots["loss_bars"]
        group = pcfg.get("group", "flow")
        metric = pcfg.get("metric", "loss_pct")
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = sorted(dfL["id"].dropna().unique().tolist())
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                ycol = f"{metric}_mean"
                if ycol not in d.columns:
                    continue
                outp = plots_dir / f"loss_bars_{group}_{idv}.png"
                plot_grouped_bars(d, outp, metric_base=metric, title=f"Loss% vs load ({group}={idv})")
                print(f"OK plot: {outp}")

    # 3b) throughput bars
    if plots.get("throughput_bars", {}).get("enable", False):
        pcfg = plots["throughput_bars"]
        group = pcfg.get("group", "flow")
        metric = pcfg.get("metric", f"throughput_total_mean_{thr_unit}")
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = sorted(dfL["id"].dropna().unique().tolist())
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                ycol = f"{metric}_mean"
                if ycol not in d.columns:
                    continue
                outp = plots_dir / f"throughput_bars_{group}_{idv}.png"
                plot_grouped_bars(d, outp, metric_base=metric, title=f"Throughput vs load ({group}={idv})")
                print(f"OK plot: {outp}")

    # 4) throughput timeseries multi controllers (1 curva por controlador)
    if plots.get("throughput_timeseries", {}).get("enable", False):
        pcfg = plots["throughput_timeseries"]
        group = pcfg.get("group", "flow")
        sel = pcfg.get("select", {})
        load_sel = sel.get("load")
        id_sel = sel.get("id")
        dfT = pick_ts_df(group)

        if not dfT.empty and load_sel is not None and id_sel is not None:
            df_load = dfT[dfT["load"] == load_sel].copy()
            if df_load.empty:
                print(f"WARNING: no hay timeseries para load={load_sel}")
            else:
                # define lista de IDs a plotear
                if id_sel == "all":
                    ids = sorted(df_load["id"].dropna().unique().tolist())
                elif isinstance(id_sel, list):
                    ids = id_sel
                else:
                    ids = [id_sel]

                min_reps = int(cfg["timeseries"].get("min_reps_per_t", 1))
                for one_id in ids:
                    d = df_load[df_load["id"] == one_id].copy()
                    if d.empty:
                        continue
                    outp = plots_dir / f"ts_throughput_{group}_{load_sel}_{one_id}.png"
                    min_reps = int(cfg["timeseries"].get("min_reps_per_t", 1))
                    crop_common = bool(cfg["timeseries"].get("crop_to_common_time", False))

                    plot_timeseries_multi(d, outp, metric="throughput_Mbps", min_reps_per_t=min_reps,
                                        crop_to_common_time=crop_common)
                    print(f"OK plot: {outp}")

    # 4b) throughput timeseries por carga (loop de loads e ids)
    if plots.get("throughput_timeseries_loads", {}).get("enable", False):
        pcfg = plots["throughput_timeseries_loads"]
        group = pcfg.get("group", "flow")
        sel = pcfg.get("select", {})
        loads_sel = sel.get("loads", "all")
        ids_sel = sel.get("id", "all")
        ctrl_sel = sel.get("controller", "all")
        dfT = pick_ts_df(group)

        if dfT.empty:
            print("WARNING: no hay datos de timeseries para throughput_timeseries_loads")
        else:
            avail_loads = sorted(dfT["load"].dropna().unique().tolist(), key=natural_load_key)
            if loads_sel == "all" or loads_sel is None:
                loads = avail_loads
            elif isinstance(loads_sel, list):
                loads = loads_sel
            else:
                loads = [loads_sel]

            avail_ctrls = sorted(dfT["controller"].dropna().unique().tolist())
            if ctrl_sel == "all" or ctrl_sel is None:
                ctrl_list = avail_ctrls
            elif isinstance(ctrl_sel, list):
                ctrl_list = ctrl_sel
            else:
                ctrl_list = [ctrl_sel]

            if ids_sel == "all" or ids_sel is None:
                ids_list = sorted(dfT["id"].dropna().unique().tolist())
            elif isinstance(ids_sel, list):
                ids_list = ids_sel
            else:
                ids_list = [ids_sel]

            min_reps = int(cfg["timeseries"].get("min_reps_per_t", 1))
            crop_common = bool(cfg["timeseries"].get("crop_to_common_time", False))
            ctrl_slug = slugify("-".join(ctrl_list)) if ctrl_list else "all"

            for load_iter in loads:
                for one_id in ids_list:
                    d = dfT[(dfT["load"] == load_iter) & (dfT["id"] == one_id)]
                    d = d[d["controller"].isin(ctrl_list)]
                    if d.empty:
                        continue
                    outp = plots_dir / f"ts_throughput_load_{group}_{load_iter}_{one_id}_{ctrl_slug}.png"
                    plot_timeseries_multi(d, outp, metric="throughput_Mbps", min_reps_per_t=min_reps,
                                          crop_to_common_time=crop_common)
                    print(f"OK plot: {outp}")

    # 4c) throughput timeseries comparativo: varias ids en la misma gráfica por carga
    if plots.get("throughput_timeseries_compare", {}).get("enable", False):
        pcfg = plots["throughput_timeseries_compare"]
        group = pcfg.get("group", "flow")
        sel = pcfg.get("select", {})
        loads_sel = sel.get("loads", "all")
        ids_sel = sel.get("id", "all")
        ctrl_sel = sel.get("controller", "all")
        dfT = pick_ts_df(group)

        if dfT.empty:
            print("WARNING: no hay datos de timeseries para throughput_timeseries_compare")
        else:
            avail_loads = sorted(dfT["load"].dropna().unique().tolist(), key=natural_load_key)
            if loads_sel == "all" or loads_sel is None:
                loads = avail_loads
            elif isinstance(loads_sel, list):
                loads = loads_sel
            else:
                loads = [loads_sel]

            avail_ctrls = sorted(dfT["controller"].dropna().unique().tolist())
            if ctrl_sel == "all" or ctrl_sel is None:
                ctrl_base = avail_ctrls
            elif isinstance(ctrl_sel, list):
                ctrl_base = ctrl_sel
            else:
                ctrl_base = [ctrl_sel]

            if ids_sel == "all" or ids_sel is None:
                ids_base = sorted(dfT["id"].dropna().unique().tolist())
            elif isinstance(ids_sel, list):
                ids_base = ids_sel
            else:
                ids_base = [ids_sel]

            min_reps = int(cfg["timeseries"].get("min_reps_per_t", 1))
            crop_common = bool(cfg["timeseries"].get("crop_to_common_time", False))
            show_ci = bool(pcfg.get("show_ci", True))
            smooth_cfg = pcfg.get("smooth", {})
            id_labels = pcfg.get("id_labels", {})
            labels_cfg = pcfg.get("labels", {})

            for load_iter in loads:
                df_load = dfT[dfT["load"] == load_iter]
                if df_load.empty:
                    continue

                ctrls_use = [c for c in ctrl_base if c in df_load["controller"].unique()]
                if not ctrls_use:
                    continue

                ids_use = ids_base if ids_sel != "all" else sorted(df_load["id"].dropna().unique().tolist())
                ids_use = [i for i in ids_use if i in df_load["id"].unique()]
                if not ids_use:
                    continue

                outp = plots_dir / f"ts_throughput_compare_{group}_{load_iter}_{slugify('-'.join(ctrls_use))}.png"
                plot_timeseries_compare_ids(df_load, outp, ids_use, ctrls_use, id_labels, labels_cfg,
                                            show_ci, min_reps, crop_common, smooth_cfg)
                print(f"OK plot: {outp}")



if __name__ == "__main__":
    main()
