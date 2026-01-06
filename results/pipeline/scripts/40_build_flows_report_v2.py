#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from fnmatch import fnmatch
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
    - si read_csv() produce 1 columna, intenta sep='\t'
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")
    return df

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def unit_throughput_factor(units: str) -> float:
    # bytes/s -> Mbps or Kbps
    if units.lower() == "mbps":
        return 8.0 / 1e6
    if units.lower() == "kbps":
        return 8.0 / 1e3
    return 8.0 / 1e6  # default Mbps

def t_critical(level: float, df: int) -> float:
    alpha = 1.0 - level
    p = 1.0 - alpha / 2.0
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(p, df))
    except Exception:
        return 1.96

def compute_ci(mean: float, std: float, n: int, level: float, method: str) -> Tuple[Optional[float], Optional[float]]:
    if n < 2 or mean is None or std is None or pd.isna(mean) or pd.isna(std):
        return (None, None)
    se = std / math.sqrt(n)
    crit = t_critical(level, n - 1) if method == "t" else 1.96
    return (mean - crit * se, mean + crit * se)

def fmt_template(tpl: str, **kwargs) -> str:
    try:
        return tpl.format(**kwargs)
    except Exception:
        return tpl


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
# Schema normalization (CSV viejo y nuevo)
# -------------------------

_PORT_TO_PROTO = {
    1883: "mqtt",
    8883: "mqtts",
    5683: "coap",
    5684: "coaps",
    5672: "amqp",
    80: "http",
    443: "https",
    554: "rtsp",
    5004: "rtp",
    5060: "sip",
    5061: "sips",
}

def infer_protocol_from_endpoints(endpoints: object) -> Optional[str]:
    if endpoints is None or (isinstance(endpoints, float) and pd.isna(endpoints)):
        return None
    s = str(endpoints)
    ports = re.findall(r":(\d{1,5})\b", s)
    for p in ports:
        try:
            proto = _PORT_TO_PROTO.get(int(p))
            if proto:
                return proto
        except Exception:
            continue
    return None

def normalize_flow_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Acepta ambos esquemas:
      - v1: count, mean, jitter_count, jitter_mean
      - v2: delay_count, delay_sum, delay_mean (+ opcionales jitter_count, jitter_sum, jitter_mean)
    y crea/normaliza columnas esperadas por el pipeline.
    """
    df = df_raw.copy()

    for col in [
        "flow", "protocol", "transport", "endpoints",
        "t_sec",
        "count", "mean",
        "delay_count", "delay_sum", "delay_mean",
        "count_a", "count_b", "lost", "throughput_bytes",
        "jitter_count", "jitter_sum", "jitter_mean",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # delay: v2 -> v1
    if df["count"].isna().all() and df["delay_count"].notna().any():
        df["count"] = df["delay_count"]
    if df["mean"].isna().all() and df["delay_mean"].notna().any():
        df["mean"] = df["delay_mean"]

    dc = pd.to_numeric(df["delay_count"], errors="coerce")
    ds = pd.to_numeric(df["delay_sum"], errors="coerce")
    if df["mean"].isna().all() and ds.notna().any() and dc.notna().any():
        with pd.option_context("mode.use_inf_as_na", True):
            df["mean"] = ds / dc.replace({0: pd.NA})

    # jitter: compute mean from sum/count if needed
    jc = pd.to_numeric(df["jitter_count"], errors="coerce")
    js = pd.to_numeric(df["jitter_sum"], errors="coerce")
    if df["jitter_mean"].isna().all() and js.notna().any() and jc.notna().any():
        with pd.option_context("mode.use_inf_as_na", True):
            df["jitter_mean"] = js / jc.replace({0: pd.NA})

    # infer protocol if missing
    if df["protocol"].isna().all() and df["endpoints"].notna().any():
        proto = infer_protocol_from_endpoints(df["endpoints"].dropna().iloc[0])
        if proto:
            df["protocol"] = proto

    return df


# -------------------------
# Protocol grouping via YAML
# -------------------------

def resolve_protocol(flow_id: str, df_norm: pd.DataFrame, cfg: dict) -> str:
    """
    Orden:
      1) protocol_groups del YAML (si hay match por nombre de CSV: flow_id)
      2) columna protocol del CSV
      3) inferencia por endpoints
      4) "unknown"
    """
    pg = cfg.get("protocol_groups") or {}
    if isinstance(pg, dict) and pg:
        for group_name, spec in pg.items():
            if not isinstance(spec, dict):
                continue
            patterns = spec.get("flows", [])
            for pat in patterns or []:
                if fnmatch(flow_id, str(pat)):
                    return str(group_name)

    if "protocol" in df_norm.columns and df_norm["protocol"].notna().any():
        return str(df_norm["protocol"].dropna().astype(str).iloc[0])

    if "endpoints" in df_norm.columns and df_norm["endpoints"].notna().any():
        proto = infer_protocol_from_endpoints(df_norm["endpoints"].dropna().iloc[0])
        if proto:
            return proto

    return "unknown"


# -------------------------
# Core compute per file/rep
# -------------------------

def compute_file_rep_summaries(
    df_norm: pd.DataFrame,
    cfg: dict,
    flow_id: str,
    protocol: str,
) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    filters = cfg["filters"]
    warm = cfg["warmup"]
    loss_cfg = cfg["loss"]
    thr_cfg = cfg["throughput"]
    units = cfg["units"]

    thr_factor = unit_throughput_factor(units.get("throughput", "Mbps"))

    df = df_norm.copy()
    df["t_sec"] = to_num(df["t_sec"])
    df = df.dropna(subset=["t_sec"]).sort_values("t_sec")
    if df.empty:
        return {}, {}, pd.DataFrame(), pd.DataFrame()

    t0 = float(df["t_sec"].iloc[0])
    df["t_rel_s"] = (df["t_sec"] - t0).astype(int)

    # warmup
    warmup_s = 0
    if warm.get("mode", "none") == "fixed":
        warmup_s = int(warm.get("fixed_seconds", 0))
    elif warm.get("mode", "none") == "auto":
        w = detect_warmup_count(df[["t_rel_s", "count"]].copy(), warm.get("auto", {}))
        warmup_s = int(warm.get("fallback_if_not_found", 0)) if w is None else int(w)

    # numeric
    df["count"] = to_num(df["count"]).fillna(0)
    df["mean"] = to_num(df["mean"])
    df["throughput_bytes"] = to_num(df["throughput_bytes"]).fillna(0)

    # used rows
    df_used = df[df["t_rel_s"] >= warmup_s].copy()
    min_count = int(filters.get("min_count_per_row", 1))
    df_used = df_used[df_used["count"] >= min_count].copy()
    if filters.get("drop_rows_with_nan_mean", True):
        df_used = df_used[df_used["mean"].notna()].copy()

    # loss
    sent_basis = loss_cfg.get("sent_basis", "count_a")
    if sent_basis not in df.columns:
        sent_basis = "count"
    df_used["sent"] = to_num(df_used.get(sent_basis, pd.Series(dtype=float))).fillna(df_used["count"])
    df_used["lost"] = to_num(df_used.get("lost", pd.Series(dtype=float))).fillna(0)
    include_neg = bool(loss_cfg.get("include_negative_loss", False))
    lost_adj = df_used["lost"] if include_neg else df_used["lost"].clip(lower=0)

    # jitter
    df_used["jitter_mean"] = to_num(df_used.get("jitter_mean", pd.Series(dtype=float)))
    df_used["jitter_count"] = to_num(df_used.get("jitter_count", pd.Series(dtype=float))).fillna(0)
    has_jitter = df_used["jitter_mean"].notna().any() and (df_used["jitter_count"].sum() > 0)

    # delay weighted mean (prefer delay_sum if present)
    delay_count = float(df_used["count"].sum())
    delay_sum_s = None
    if "delay_sum" in df_used.columns and df_used["delay_sum"].notna().any():
        delay_sum_s = float(to_num(df_used["delay_sum"]).fillna(0).sum())
    if delay_sum_s is not None and delay_count > 0:
        delay_mean_s = delay_sum_s / delay_count
    else:
        delay_weight = float((df_used["mean"] * df_used["count"]).sum()) if delay_count > 0 else None
        delay_mean_s = (delay_weight / delay_count) if (delay_count and delay_weight is not None) else None
    delay_mean_ms = (delay_mean_s * 1000.0) if delay_mean_s is not None else None

    # jitter weighted mean (prefer jitter_sum)
    jitter_mean_ms = None
    jitter_count_total = float(df_used["jitter_count"].sum()) if has_jitter else 0.0
    if has_jitter:
        jitter_sum_s = None
        if "jitter_sum" in df_used.columns and df_used["jitter_sum"].notna().any():
            jitter_sum_s = float(to_num(df_used["jitter_sum"]).fillna(0).sum())
        if jitter_sum_s is not None and jitter_count_total > 0:
            jm_s = jitter_sum_s / jitter_count_total
        else:
            jw = float((df_used["jitter_mean"] * df_used["jitter_count"]).sum()) if jitter_count_total > 0 else None
            jm_s = (jw / jitter_count_total) if (jitter_count_total and jw is not None) else None
        jitter_mean_ms = (jm_s * 1000.0) if jm_s is not None else None

    # throughput mean
    total_bytes = float(df_used["throughput_bytes"].sum())
    if thr_cfg.get("normalization", "wall") == "active":
        denom_s = max(1, int(df_used.shape[0]))
    else:
        denom_s = int(df_used["t_rel_s"].max() - df_used["t_rel_s"].min() + 1) if df_used.shape[0] >= 2 else 1
    throughput_mean = (total_bytes / denom_s) * thr_factor
    thr_unit = cfg["units"].get("throughput", "Mbps")

    # loss %
    sent_total = float(df_used["sent"].sum())
    lost_total = float(lost_adj.sum()) if not df_used.empty else 0.0
    loss_pct = (100.0 * lost_total / sent_total) if sent_total > 0 else None

    flow_summary = {
        "id": flow_id,
        "protocol": protocol,
        "warmup_s": int(warmup_s),

        "delay_mean_ms": delay_mean_ms,
        "delay_weight_count": delay_count,

        "jitter_mean_ms": jitter_mean_ms,
        "jitter_weight_count": jitter_count_total,

        f"throughput_total_mean_{thr_unit}": throughput_mean,
        "throughput_total_bytes": total_bytes,
        "throughput_denom_s": float(denom_s),

        "lost_total_pkts": lost_total,
        "sent_total_pkts": sent_total,
        "loss_pct": loss_pct,
        "rows_used": int(df_used.shape[0]),
    }

    proto_accum = {
        "protocol": protocol,
        "delay_weight_sum_s": float((df_used["mean"] * df_used["count"]).sum()) if delay_count > 0 else 0.0,
        "delay_sum_s": float(to_num(df_used.get("delay_sum", pd.Series(dtype=float))).fillna(0).sum()) if "delay_sum" in df_used.columns else 0.0,
        "delay_count_sum": float(delay_count),

        "jitter_weight_sum_s": float((df_used["jitter_mean"] * df_used["jitter_count"]).sum()) if has_jitter else 0.0,
        "jitter_sum_s": float(to_num(df_used.get("jitter_sum", pd.Series(dtype=float))).fillna(0).sum()) if "jitter_sum" in df_used.columns else 0.0,
        "jitter_count_sum": float(jitter_count_total) if has_jitter else 0.0,

        "throughput_total_bytes": total_bytes,
        "t_sec_min": float(df_used["t_sec"].min()) if not df_used.empty else None,
        "t_sec_max": float(df_used["t_sec"].max()) if not df_used.empty else None,
        "lost_total_pkts": lost_total,
        "sent_total_pkts": sent_total,
    }

    # timeseries flow (Mbps)
    df_ts = df.copy() if cfg["timeseries"].get("include_warmup", True) else df[df["t_rel_s"] >= warmup_s].copy()
    df_ts = df_ts[["t_rel_s", "throughput_bytes"]].copy()
    df_ts["throughput_Mbps"] = df_ts["throughput_bytes"] * unit_throughput_factor("Mbps")
    df_ts = df_ts.drop(columns=["throughput_bytes"])

    # timeseries proto rows (bytes/s + protocol label)
    df_ts_proto_rows = df.copy() if cfg["timeseries"].get("include_warmup", True) else df[df["t_rel_s"] >= warmup_s].copy()
    df_ts_proto_rows = df_ts_proto_rows[["t_sec", "throughput_bytes"]].copy()
    df_ts_proto_rows["protocol"] = protocol

    return flow_summary, proto_accum, df_ts, df_ts_proto_rows


# -------------------------
# Plot helpers (con estilo)
# -------------------------

def _apply_axes_style(ax, style: dict, defaults: dict):
    xlabel = style.get("xlabel", defaults.get("xlabel"))
    ylabel = style.get("ylabel", defaults.get("ylabel"))
    title = style.get("title", defaults.get("title"))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

def plot_lines_vs_load(df: pd.DataFrame, out_path: Path, metric_base: str, style: dict, ctx: dict):
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
        d["__k"] = d["load"].apply(natural_load_key)
        d = d.sort_values("__k")

        y, lo, hi = [], [], []
        for ld in loads:
            row = d[d["load"] == ld]
            if row.empty:
                y.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))
            else:
                y.append(row.iloc[0].get(mean_col, float("nan")))
                lo.append(row.iloc[0].get(low_col, float("nan")))
                hi.append(row.iloc[0].get(high_col, float("nan")))

        ax.plot(x, y, marker="o", label=ctrl)

        lo_s = pd.to_numeric(pd.Series(lo), errors="coerce")
        hi_s = pd.to_numeric(pd.Series(hi), errors="coerce")
        y_s  = pd.to_numeric(pd.Series(y),  errors="coerce")
        mask = lo_s.notna() & hi_s.notna() & y_s.notna()
        if mask.any():
            ax.fill_between(pd.Series(x)[mask], lo_s[mask], hi_s[mask], alpha=0.15)

    legend_title = style.get("legend_title")
    ax.legend(title=str(legend_title) if legend_title else None)

    defaults = {"xlabel": "load", "ylabel": metric_base, "title": f"{metric_base} vs load"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    _apply_axes_style(ax, style2, defaults)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_grouped_bars(df: pd.DataFrame, out_path: Path, metric_base: str, style: dict, ctx: dict):
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
        ys, yerr_low, yerr_high = [], [], []
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

    legend_title = style.get("legend_title")
    ax.legend(title=str(legend_title) if legend_title else None)

    defaults = {"xlabel": "load", "ylabel": metric_base, "title": f"{metric_base} vs load"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    _apply_axes_style(ax, style2, defaults)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_multi_controllers(ts_df: pd.DataFrame, out_path: Path, metric: str,
                                      min_reps_per_t: int, crop_to_common_time: bool,
                                      style: dict, ctx: dict):
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

    if crop_to_common_time:
        tmax_by_ctrl = df.groupby("controller")["t_rel_s"].max().dropna()
        if not tmax_by_ctrl.empty:
            common_tmax = int(tmax_by_ctrl.min())
            df = df[df["t_rel_s"] <= common_tmax]

    fig, ax = plt.subplots()
    for ctrl in sorted(df["controller"].dropna().unique().tolist()):
        d = df[df["controller"] == ctrl].sort_values("t_rel_s")
        ax.plot(d["t_rel_s"], d[mean_col], label=ctrl)
        dd = d.dropna(subset=[low_col, high_col, mean_col])
        if not dd.empty:
            ax.fill_between(dd["t_rel_s"], dd[low_col], dd[high_col], alpha=0.15)

    legend_title = style.get("legend_title")
    ax.legend(title=str(legend_title) if legend_title else None)

    defaults = {"xlabel": "t_rel (s)", "ylabel": metric, "title": f"{metric} vs tiempo"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    _apply_axes_style(ax, style2, defaults)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_multi_loads(ts_df: pd.DataFrame, out_path: Path, metric: str,
                                min_reps_per_t: int, crop_to_common_time: bool,
                                style: dict, ctx: dict):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"
    count_col = f"{metric}_count"

    df = ts_df.copy().sort_values(["load", "t_rel_s"])
    if count_col in df.columns:
        df = df[df[count_col] >= min_reps_per_t]
    df = df.dropna(subset=[mean_col])
    if df.empty:
        print(f"WARNING: serie vacía -> {out_path}")
        return

    if crop_to_common_time:
        tmax_by_load = df.groupby("load")["t_rel_s"].max().dropna()
        if not tmax_by_load.empty:
            common_tmax = int(tmax_by_load.min())
            df = df[df["t_rel_s"] <= common_tmax]

    fig, ax = plt.subplots()
    loads = sorted(df["load"].dropna().unique().tolist(), key=natural_load_key)
    for ld in loads:
        d = df[df["load"] == ld].sort_values("t_rel_s")
        ax.plot(d["t_rel_s"], d[mean_col], label=str(ld))
        dd = d.dropna(subset=[low_col, high_col, mean_col])
        if not dd.empty:
            ax.fill_between(dd["t_rel_s"], dd[low_col], dd[high_col], alpha=0.15)

    legend_title = style.get("legend_title", "Carga")
    ax.legend(title=str(legend_title) if legend_title else None)

    defaults = {"xlabel": "t_rel (s)", "ylabel": metric, "title": f"{metric} vs tiempo"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    _apply_axes_style(ax, style2, defaults)

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

    # defaults
    cfg.setdefault("filters", {})
    cfg.setdefault("warmup", {"mode": "none"})
    cfg.setdefault("loss", {})
    cfg.setdefault("throughput", {})
    cfg.setdefault("timeseries", {})
    cfg.setdefault("units", {"throughput": "Mbps"})
    cfg.setdefault("stats", {"confidence": {"enable": True, "level": 0.95, "method": "t"}})

    extracted_root = Path(cfg["inputs"]["extracted_root"]).resolve()
    flows_glob = cfg["inputs"]["flows_glob"]
    flow_filter = cfg["inputs"].get("flow_filter", []) or []
    group_by = cfg["inputs"].get("group_by", ["flow"])
    thr_unit = cfg["units"].get("throughput", "Mbps")

    out_dir = Path(cfg["outputs"]["out_dir"]).resolve()
    plots_dir = Path(cfg["outputs"]["plots_dir"]).resolve()
    safe_mkdir(out_dir)
    safe_mkdir(plots_dir)

    p_rep_flow = out_dir / cfg["outputs"]["csv"]["rep_summary_flow"]
    p_rep_proto = out_dir / cfg["outputs"]["csv"]["rep_summary_protocol"]
    p_load_flow = out_dir / cfg["outputs"]["csv"]["load_summary_flow"]
    p_load_proto = out_dir / cfg["outputs"]["csv"]["load_summary_protocol"]
    p_ts_flow = out_dir / cfg["outputs"]["csv"]["ts_throughput_flow"]
    p_ts_proto = out_dir / cfg["outputs"]["csv"]["ts_throughput_protocol"]
    p_missing = out_dir / cfg["outputs"]["csv"]["missing"]

    required_csvs = [p_rep_flow, p_rep_proto, p_load_flow, p_load_proto, p_ts_flow, p_ts_proto]
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
        missing: List[dict] = []
        rep_flow_rows: List[dict] = []
        rep_proto_acc: Dict[Tuple[str, str, str, str], dict] = {}

        ts_flow_rows: List[dict] = []
        ts_proto_rows_raw: List[dict] = []

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

                    df_norm = normalize_flow_schema(df_raw)
                    proto = resolve_protocol(flow_id, df_norm, cfg)

                    try:
                        flow_summary, proto_accum, df_ts, df_ts_proto = compute_file_rep_summaries(df_norm, cfg, flow_id, proto)
                    except Exception as e:
                        missing.append({"controller": controller, "load": load, "rep": rep, "file": str(csv_path), "reason": f"compute_error:{e}"})
                        continue
                    if not flow_summary:
                        missing.append({"controller": controller, "load": load, "rep": rep, "file": str(csv_path), "reason": "no_valid_rows_after_filter"})
                        continue

                    rep_flow_rows.append({"controller": controller, "load": load, "rep": rep, **flow_summary})

                    # protocol rep aggregation
                    k = (controller, load, rep, proto)
                    if k not in rep_proto_acc:
                        rep_proto_acc[k] = {
                            "controller": controller, "load": load, "rep": rep,
                            "id": proto, "protocol": proto,
                            "delay_weight_sum_s": 0.0, "delay_sum_s": 0.0, "delay_count_sum": 0.0,
                            "jitter_weight_sum_s": 0.0, "jitter_sum_s": 0.0, "jitter_count_sum": 0.0,
                            "throughput_total_bytes": 0.0,
                            "lost_total_pkts": 0.0, "sent_total_pkts": 0.0,
                            "t_sec_min": None, "t_sec_max": None,
                        }
                    acc = rep_proto_acc[k]
                    acc["delay_weight_sum_s"] += float(proto_accum.get("delay_weight_sum_s", 0.0))
                    acc["delay_sum_s"] += float(proto_accum.get("delay_sum_s", 0.0))
                    acc["delay_count_sum"] += float(proto_accum.get("delay_count_sum", 0.0))
                    acc["jitter_weight_sum_s"] += float(proto_accum.get("jitter_weight_sum_s", 0.0))
                    acc["jitter_sum_s"] += float(proto_accum.get("jitter_sum_s", 0.0))
                    acc["jitter_count_sum"] += float(proto_accum.get("jitter_count_sum", 0.0))
                    acc["throughput_total_bytes"] += float(proto_accum.get("throughput_total_bytes", 0.0))
                    acc["lost_total_pkts"] += float(proto_accum.get("lost_total_pkts", 0.0))
                    acc["sent_total_pkts"] += float(proto_accum.get("sent_total_pkts", 0.0))
                    tmin = proto_accum.get("t_sec_min")
                    tmax = proto_accum.get("t_sec_max")
                    if tmin is not None:
                        acc["t_sec_min"] = float(tmin) if acc["t_sec_min"] is None else min(float(acc["t_sec_min"]), float(tmin))
                    if tmax is not None:
                        acc["t_sec_max"] = float(tmax) if acc["t_sec_max"] is None else max(float(acc["t_sec_max"]), float(tmax))

                    # rep flow timeseries
                    for _, r in df_ts.iterrows():
                        ts_flow_rows.append({
                            "controller": controller, "load": load, "rep": rep,
                            "id": flow_id, "t_rel_s": int(r["t_rel_s"]),
                            "throughput_Mbps": float(r["throughput_Mbps"]) if pd.notna(r["throughput_Mbps"]) else None,
                        })

                    # rep protocol raw rows
                    for _, r in df_ts_proto.iterrows():
                        ts_proto_rows_raw.append({
                            "controller": controller, "load": load, "rep": rep,
                            "protocol": proto,
                            "t_sec": float(r["t_sec"]) if pd.notna(r["t_sec"]) else None,
                            "throughput_bytes": float(r["throughput_bytes"]) if pd.notna(r["throughput_bytes"]) else 0.0,
                        })

        rep_flow_df = pd.DataFrame(rep_flow_rows)
        rep_proto_df = pd.DataFrame(list(rep_proto_acc.values()))
        miss_df = pd.DataFrame(missing)

        # confidence settings
        ci_cfg = (cfg.get("stats", {}) or {}).get("confidence", {}) or {}
        ci_enable = bool(ci_cfg.get("enable", True))
        ci_level = float(ci_cfg.get("level", 0.95))
        ci_method = str(ci_cfg.get("method", "t"))

        def summarize_over_reps(df_rep: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
            if df_rep.empty:
                return pd.DataFrame()
            rows = []
            for (controller, load, idv), g in df_rep.groupby(["controller", "load", "id"]):
                row = {"controller": controller, "load": load, "id": idv}
                for m in metric_cols:
                    if m not in g.columns:
                        continue
                    vals = pd.to_numeric(g[m], errors="coerce").dropna()
                    n = int(vals.shape[0])
                    mean = float(vals.mean()) if n > 0 else None
                    std = float(vals.std(ddof=1)) if n > 1 else None
                    row[f"{m}_mean"] = mean
                    row[f"{m}_std"] = std
                    row[f"{m}_n"] = n
                    if ci_enable and n >= 2 and mean is not None and std is not None:
                        lo, hi = compute_ci(mean, std, n, ci_level, ci_method)
                        row[f"{m}_ci95_low"] = lo
                        row[f"{m}_ci95_high"] = hi
                    else:
                        row[f"{m}_ci95_low"] = None
                        row[f"{m}_ci95_high"] = None
                rows.append(row)
            out = pd.DataFrame(rows)
            if not out.empty:
                out["__k"] = out["load"].apply(natural_load_key)
                out = out.sort_values(["id", "controller", "__k"]).drop(columns=["__k"])
            return out

        thr_metric = f"throughput_total_mean_{thr_unit}"
        flow_metrics = ["delay_mean_ms", "jitter_mean_ms", thr_metric, "loss_pct"]
        proto_metrics = ["delay_mean_ms", "jitter_mean_ms", thr_metric, "loss_pct"]

        load_flow_df = summarize_over_reps(rep_flow_df, flow_metrics) if "flow" in group_by else pd.DataFrame()
        load_proto_df = summarize_over_reps(rep_proto_df, proto_metrics) if "protocol" in group_by else pd.DataFrame()

        # timeseries mean/CI over reps (flow)
        ts_flow_mean_df = pd.DataFrame()
        if ts_flow_rows:
            df_ts_rep = pd.DataFrame(ts_flow_rows)
            out = []
            for (controller, load, idv, t), g in df_ts_rep.groupby(["controller", "load", "id", "t_rel_s"]):
                vals = pd.to_numeric(g["throughput_Mbps"], errors="coerce").dropna()
                n = int(vals.shape[0])
                mean = float(vals.mean()) if n > 0 else None
                std = float(vals.std(ddof=1)) if n > 1 else None
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
            ts_flow_mean_df = pd.DataFrame(out)
            if not ts_flow_mean_df.empty:
                ts_flow_mean_df["__k"] = ts_flow_mean_df["load"].apply(natural_load_key)
                ts_flow_mean_df = ts_flow_mean_df.sort_values(["id", "controller", "__k", "t_rel_s"]).drop(columns=["__k"])

        # timeseries protocol: sum bytes/s per second per rep+protocol, then mean/CI over reps
        ts_proto_mean_df = pd.DataFrame()
        if ts_proto_rows_raw:
            df_ts_raw = pd.DataFrame(ts_proto_rows_raw).dropna(subset=["t_sec"])
            rep_rows = []
            for (controller, load, rep, proto), g in df_ts_raw.groupby(["controller", "load", "rep", "protocol"]):
                g = g.sort_values("t_sec").copy()
                t0 = float(g["t_sec"].iloc[0])
                g["t_rel_s"] = (g["t_sec"] - t0).astype(int)
                gg = g.groupby("t_rel_s")["throughput_bytes"].sum().reset_index()
                gg["controller"] = controller
                gg["load"] = load
                gg["rep"] = rep
                gg["id"] = proto
                gg["throughput_Mbps"] = gg["throughput_bytes"] * unit_throughput_factor("Mbps")
                rep_rows.append(gg[["controller", "load", "rep", "id", "t_rel_s", "throughput_Mbps"]])
            if rep_rows:
                df_rep = pd.concat(rep_rows, ignore_index=True)
                out = []
                for (controller, load, idv, t), g in df_rep.groupby(["controller", "load", "id", "t_rel_s"]):
                    vals = pd.to_numeric(g["throughput_Mbps"], errors="coerce").dropna()
                    n = int(vals.shape[0])
                    mean = float(vals.mean()) if n > 0 else None
                    std = float(vals.std(ddof=1)) if n > 1 else None
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

        # save CSVs
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

    # plots
    plots = cfg.get("plots", {}) or {}

    def pick_load_df(group: str):
        return load_flow_df if group == "flow" else load_proto_df

    def pick_ts_df(group: str):
        return ts_flow_mean_df if group == "flow" else ts_proto_mean_df

    def select_list(val, universe: List[str]) -> List[str]:
        if val == "all" or val is None:
            return universe
        if isinstance(val, list):
            return [str(x) for x in val]
        return [str(val)]

    # delay vs load
    if plots.get("delay_vs_load", {}).get("enable", False):
        pcfg = plots["delay_vs_load"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", "delay_mean_ms"))
        style = pcfg.get("style", {}) or {}
        dfL = pick_load_df(group)
        if not dfL.empty:
            for idv in sorted(dfL["id"].dropna().unique().tolist()):
                d = dfL[dfL["id"] == idv].copy()
                if f"{metric}_mean" not in d.columns:
                    continue
                outp = plots_dir / f"delay_vs_load_{group}_{idv}.png"
                plot_lines_vs_load(d, outp, metric, style, {"id": idv, "group": group, "metric": metric})
                print(f"OK plot: {outp}")

    # jitter vs load
    if plots.get("jitter_vs_load", {}).get("enable", False):
        pcfg = plots["jitter_vs_load"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", "jitter_mean_ms"))
        style = pcfg.get("style", {}) or {}
        dfL = pick_load_df(group)
        if not dfL.empty:
            for idv in sorted(dfL["id"].dropna().unique().tolist()):
                d = dfL[dfL["id"] == idv].copy()
                ycol = f"{metric}_mean"
                if ycol not in d.columns:
                    continue
                if (cfg.get("jitter", {}) or {}).get("plot_only_if_present", True) and d[ycol].dropna().empty:
                    continue
                outp = plots_dir / f"jitter_vs_load_{group}_{idv}.png"
                plot_lines_vs_load(d, outp, metric, style, {"id": idv, "group": group, "metric": metric})
                print(f"OK plot: {outp}")

    # loss bars
    if plots.get("loss_bars", {}).get("enable", False):
        pcfg = plots["loss_bars"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", "loss_pct"))
        style = pcfg.get("style", {}) or {}
        dfL = pick_load_df(group)
        if not dfL.empty:
            for idv in sorted(dfL["id"].dropna().unique().tolist()):
                d = dfL[dfL["id"] == idv].copy()
                if f"{metric}_mean" not in d.columns:
                    continue
                outp = plots_dir / f"loss_bars_{group}_{idv}.png"
                plot_grouped_bars(d, outp, metric, style, {"id": idv, "group": group, "metric": metric})
                print(f"OK plot: {outp}")

    # throughput bars
    if plots.get("throughput_bars", {}).get("enable", False):
        pcfg = plots["throughput_bars"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", f"throughput_total_mean_{thr_unit}"))
        style = pcfg.get("style", {}) or {}
        dfL = pick_load_df(group)
        if not dfL.empty:
            for idv in sorted(dfL["id"].dropna().unique().tolist()):
                d = dfL[dfL["id"] == idv].copy()
                if f"{metric}_mean" not in d.columns:
                    continue
                outp = plots_dir / f"throughput_bars_{group}_{idv}.png"
                plot_grouped_bars(d, outp, metric, style, {"id": idv, "group": group, "metric": metric})
                print(f"OK plot: {outp}")

    # throughput timeseries for one load (curves = controllers)
    if plots.get("throughput_timeseries", {}).get("enable", False):
        pcfg = plots["throughput_timeseries"]
        group = str(pcfg.get("group", "flow"))
        style = pcfg.get("style", {}) or {}
        sel = pcfg.get("select", {}) or {}
        load_sel = sel.get("load")
        id_sel = sel.get("id")
        controller_sel = sel.get("controller", "all")

        dfT = pick_ts_df(group)
        if not dfT.empty and load_sel is not None and id_sel is not None:
            df_load = dfT[dfT["load"] == load_sel].copy()
            if not df_load.empty:
                ids = select_list(id_sel, sorted(df_load["id"].dropna().unique().tolist()))
                ctrls = select_list(controller_sel, sorted(df_load["controller"].dropna().unique().tolist()))
                min_reps = int(cfg["timeseries"].get("min_reps_per_t", 1))
                crop_common = bool(cfg["timeseries"].get("crop_to_common_time", False))

                for one_id in ids:
                    d0 = df_load[(df_load["id"] == one_id) & (df_load["controller"].isin(ctrls))].copy()
                    if d0.empty:
                        continue
                    outp = plots_dir / f"ts_throughput_{group}_load_{load_sel}_{one_id}.png"
                    plot_timeseries_multi_controllers(
                        d0, outp, "throughput_Mbps",
                        min_reps, crop_common,
                        style, {"id": one_id, "group": group, "load": load_sel, "metric": "throughput_Mbps"}
                    )
                    print(f"OK plot: {outp}")

    # NEW: throughput timeseries by load (curves = loads)
    if plots.get("throughput_timeseries_loads", {}).get("enable", False):
        pcfg = plots["throughput_timeseries_loads"]
        group = str(pcfg.get("group", "flow"))
        style = pcfg.get("style", {}) or {}
        sel = pcfg.get("select", {}) or {}
        id_sel = sel.get("id", "all")
        controller_sel = sel.get("controller", "all")
        loads_sel = sel.get("loads", "all")

        dfT = pick_ts_df(group)
        if not dfT.empty:
            ids = select_list(id_sel, sorted(dfT["id"].dropna().unique().tolist()))
            ctrls = select_list(controller_sel, sorted(dfT["controller"].dropna().unique().tolist()))
            loads = select_list(loads_sel, sorted(dfT["load"].dropna().unique().tolist(), key=natural_load_key))

            min_reps = int(cfg["timeseries"].get("min_reps_per_t", 1))
            crop_common = bool(cfg["timeseries"].get("crop_to_common_time", False))

            for one_id in ids:
                for one_ctrl in ctrls:
                    d0 = dfT[(dfT["id"] == one_id) & (dfT["controller"] == one_ctrl) & (dfT["load"].isin(loads))].copy()
                    if d0.empty:
                        continue
                    outp = plots_dir / f"ts_throughput_{group}_by_load_{one_ctrl}_{one_id}.png"
                    plot_timeseries_multi_loads(
                        d0, outp, "throughput_Mbps",
                        min_reps, crop_common,
                        style, {"id": one_id, "group": group, "controller": one_ctrl, "metric": "throughput_Mbps"}
                    )
                    print(f"OK plot: {outp}")

if __name__ == "__main__":
    main()