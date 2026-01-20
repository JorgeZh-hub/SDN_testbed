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
import numpy as np


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

def format_load_label(load_val: str, suffix: Optional[str]) -> str:
    """
    Permite reemplazar el sufijo de la carga (ej. '2a' -> '2n' si suffix='n').
    """
    if not suffix:
        return str(load_val)
    s = str(load_val)
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([A-Za-z]+)?$", s)
    if m:
        return f"{m.group(1)}{suffix}"
    return f"{s}{suffix}"

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

def clamp_ci(lo: Optional[float], hi: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if lo is not None and not pd.isna(lo) and lo < 0:
        lo = 0.0
    if hi is not None and not pd.isna(hi) and hi < 0:
        hi = 0.0
    return lo, hi


def filter_outliers(rep_df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtra repeticiones outliers por métrica usando z-score (por controller/load/id).
    """
    out_cfg = (cfg.get("stats", {}) or {}).get("outliers", {}) or {}
    if not out_cfg.get("enable", False) or rep_df.empty:
        return rep_df, pd.DataFrame()

    metrics = out_cfg.get("metrics", ["jitter_mean_ms"])
    z_th = float(out_cfg.get("zscore_threshold", 3.0))

    keep_mask = pd.Series(True, index=rep_df.index)
    removed_rows = []

    for (ctrl, load, fid), g in rep_df.groupby(["controller", "load", "id"], dropna=False):
        for m in metrics:
            if m not in g.columns:
                continue
            vals = pd.to_numeric(g[m], errors="coerce")
            if vals.dropna().shape[0] < 2:
                continue
            mean = vals.mean()
            std = vals.std(ddof=1)
            if pd.isna(std) or std <= 0:
                continue
            zscores = (vals - mean) / std
            idx_bad = zscores[zscores.abs() > z_th].index
            if len(idx_bad) > 0:
                keep_mask.loc[idx_bad] = False
                for idx in idx_bad:
                    removed_rows.append({
                        "controller": ctrl,
                        "load": load,
                        "id": fid,
                        "metric": m,
                        "value": vals.loc[idx],
                        "zscore": zscores.loc[idx],
                    })

    rep_df_filt = rep_df[keep_mask].copy()
    removed_df = pd.DataFrame(removed_rows)
    return rep_df_filt, removed_df

def fmt_template(tpl: str, **kwargs) -> str:
    try:
        return tpl.format(**kwargs)
    except Exception:
        return tpl


def get_metric_label(metric_labels: dict, metric: str) -> str:
    lbl = (metric_labels or {}).get(metric)
    return str(lbl) if lbl else metric

def display_id(id_labels: dict, idv: str) -> str:
    return str(id_labels.get(idv, idv))

def clip_nonneg(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0)


def smooth_series(s: pd.Series, smooth_cfg: Optional[dict]) -> pd.Series:
    if not smooth_cfg or not smooth_cfg.get("enable", False):
        return s
    try:
        w = int(smooth_cfg.get("window_points", 1))
    except Exception:
        w = 1
    if w <= 1:
        return s
    return s.rolling(window=w, min_periods=1, center=True).mean()

def slugify(val: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(val))
    s = s.strip("_")
    return s or "all"

def safe_float(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

def log_points(out_path: Path, rows: List[str], title: Optional[str] = None):
    if not rows:
        return
    hdr = title or f"POINTS {out_path.name}"
    print(hdr)
    for line in rows:
        print(f"  {line}")


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
) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame, dict]:
    filters = cfg["filters"]
    warm = cfg["warmup"]
    loss_cfg = cfg["loss"]
    thr_cfg = cfg["throughput"]
    units = cfg["units"]
    stats_cfg = cfg.get("stats", {}) or {}
    tw_cfg = stats_cfg.get("time_window", {}) or {}
    tw_enable = bool(tw_cfg.get("enable", False))
    tw_start = safe_float(tw_cfg.get("start_s", 0.0)) or 0.0
    tw_end = safe_float(tw_cfg.get("end_s", tw_cfg.get("max_s", None)))
    tw_require_full = bool(tw_cfg.get("require_full_window", True))

    thr_factor = unit_throughput_factor(units.get("throughput", "Mbps"))

    df = df_norm.copy()
    df["t_sec"] = to_num(df["t_sec"])
    df = df.dropna(subset=["t_sec"]).sort_values("t_sec")
    if df.empty:
        return {}, {}, pd.DataFrame(), pd.DataFrame(), {}

    t0 = float(df["t_sec"].iloc[0])
    df["t_rel_s"] = (df["t_sec"] - t0).astype(int)
    t_rel_min = safe_float(df["t_rel_s"].min())
    t_rel_max = safe_float(df["t_rel_s"].max())

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

    df_metrics = df_used.copy()
    window_ok = True
    if tw_enable:
        df_metrics = df_metrics[df_metrics["t_rel_s"] >= tw_start]
        if tw_end is not None:
            df_metrics = df_metrics[df_metrics["t_rel_s"] <= tw_end]
        if df_metrics.empty:
            window_ok = False
        if tw_require_full and tw_end is not None and (t_rel_max is None or t_rel_max < tw_end):
            window_ok = False

    # loss
    sent_basis = loss_cfg.get("sent_basis", "count_a")
    if sent_basis not in df.columns:
        sent_basis = "count"
    df_metrics["sent"] = to_num(df_metrics.get(sent_basis, pd.Series(dtype=float))).fillna(df_metrics["count"])
    df_metrics["lost"] = to_num(df_metrics.get("lost", pd.Series(dtype=float))).fillna(0)
    include_neg = bool(loss_cfg.get("include_negative_loss", False))
    lost_adj = df_metrics["lost"] if include_neg else df_metrics["lost"].clip(lower=0)

    # jitter
    df_metrics["jitter_mean"] = to_num(df_metrics.get("jitter_mean", pd.Series(dtype=float)))
    df_metrics["jitter_count"] = to_num(df_metrics.get("jitter_count", pd.Series(dtype=float))).fillna(0)
    has_jitter = df_metrics["jitter_mean"].notna().any() and (df_metrics["jitter_count"].sum() > 0)

    # delay weighted mean (prefer delay_sum if present)
    delay_count = float(df_metrics["count"].sum())
    delay_sum_s = None
    if "delay_sum" in df_metrics.columns and df_metrics["delay_sum"].notna().any():
        delay_sum_s = float(to_num(df_metrics["delay_sum"]).fillna(0).sum())
    if delay_sum_s is not None and delay_count > 0:
        delay_mean_s = delay_sum_s / delay_count
    else:
        delay_weight = float((df_metrics["mean"] * df_metrics["count"]).sum()) if delay_count > 0 else None
        delay_mean_s = (delay_weight / delay_count) if (delay_count and delay_weight is not None) else None
    delay_mean_ms = (delay_mean_s * 1000.0) if delay_mean_s is not None else None

    # jitter weighted mean (prefer jitter_sum)
    jitter_mean_ms = None
    jitter_count_total = float(df_metrics["jitter_count"].sum()) if has_jitter else 0.0
    if has_jitter:
        jitter_sum_s = None
        if "jitter_sum" in df_metrics.columns and df_metrics["jitter_sum"].notna().any():
            jitter_sum_s = float(to_num(df_metrics["jitter_sum"]).fillna(0).sum())
        if jitter_sum_s is not None and jitter_count_total > 0:
            jm_s = jitter_sum_s / jitter_count_total
        else:
            jw = float((df_metrics["jitter_mean"] * df_metrics["jitter_count"]).sum()) if jitter_count_total > 0 else None
            jm_s = (jw / jitter_count_total) if (jitter_count_total and jw is not None) else None
        jitter_mean_ms = (jm_s * 1000.0) if jm_s is not None else None

    # throughput mean
    total_bytes = float(df_metrics["throughput_bytes"].sum())
    if thr_cfg.get("normalization", "wall") == "active":
        denom_s = max(1, int(df_metrics.shape[0]))
    else:
        denom_s = int(df_metrics["t_rel_s"].max() - df_metrics["t_rel_s"].min() + 1) if df_metrics.shape[0] >= 2 else 1
    throughput_mean = (total_bytes / denom_s) * thr_factor
    thr_unit = cfg["units"].get("throughput", "Mbps")

    # loss %
    sent_total = float(df_metrics["sent"].sum())
    lost_total = float(lost_adj.sum()) if not df_metrics.empty else 0.0
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
        "rows_used": int(df_metrics.shape[0]),
    }

    proto_accum = {
        "protocol": protocol,
        "delay_weight_sum_s": float((df_metrics["mean"] * df_metrics["count"]).sum()) if delay_count > 0 else 0.0,
        "delay_sum_s": float(to_num(df_metrics.get("delay_sum", pd.Series(dtype=float))).fillna(0).sum()) if "delay_sum" in df_metrics.columns else 0.0,
        "delay_count_sum": float(delay_count),

        "jitter_weight_sum_s": float((df_metrics["jitter_mean"] * df_metrics["jitter_count"]).sum()) if has_jitter else 0.0,
        "jitter_sum_s": float(to_num(df_metrics.get("jitter_sum", pd.Series(dtype=float))).fillna(0).sum()) if "jitter_sum" in df_metrics.columns else 0.0,
        "jitter_count_sum": float(jitter_count_total) if has_jitter else 0.0,

        "throughput_total_bytes": total_bytes,
        "t_sec_min": float(df_metrics["t_sec"].min()) if not df_metrics.empty else None,
        "t_sec_max": float(df_metrics["t_sec"].max()) if not df_metrics.empty else None,
        "lost_total_pkts": lost_total,
        "sent_total_pkts": sent_total,
    }

    # timeseries flow (Mbps)
    df_ts = df.copy() if cfg["timeseries"].get("include_warmup", True) else df[df["t_rel_s"] >= warmup_s].copy()
    if tw_enable:
        df_ts = df_ts[df_ts["t_rel_s"] >= tw_start]
        if tw_end is not None:
            df_ts = df_ts[df_ts["t_rel_s"] <= tw_end]
    df_ts = df_ts[["t_rel_s", "throughput_bytes"]].copy()
    df_ts["throughput_Mbps"] = df_ts["throughput_bytes"] * unit_throughput_factor("Mbps")
    df_ts = df_ts.drop(columns=["throughput_bytes"])

    # timeseries proto rows (bytes/s + protocol label)
    df_ts_proto_rows = df.copy() if cfg["timeseries"].get("include_warmup", True) else df[df["t_rel_s"] >= warmup_s].copy()
    if tw_enable:
        df_ts_proto_rows = df_ts_proto_rows[df_ts_proto_rows["t_rel_s"] >= tw_start]
        if tw_end is not None:
            df_ts_proto_rows = df_ts_proto_rows[df_ts_proto_rows["t_rel_s"] <= tw_end]
    df_ts_proto_rows = df_ts_proto_rows[["t_sec", "throughput_bytes"]].copy()
    df_ts_proto_rows["protocol"] = protocol
    time_meta = {
        "id": flow_id,
        "t_rel_min": t_rel_min,
        "t_rel_max": t_rel_max,
        "duration_s": (t_rel_max - t_rel_min) if (t_rel_max is not None and t_rel_min is not None) else None,
        "rows_raw": int(df.shape[0]),
        "rows_after_filters": int(df_used.shape[0]),
        "rows_after_time_window": int(df_metrics.shape[0]),
        "warmup_s": warmup_s,
        "time_window_start_s": tw_start if tw_enable else None,
        "time_window_end_s": tw_end if tw_enable else None,
        "window_ok": bool(window_ok),
    }

    return flow_summary, proto_accum, df_ts, df_ts_proto_rows, time_meta


# -------------------------
# Plot helpers (con estilo)
# -------------------------

def _apply_axes_style(ax, style: dict, defaults: dict):
    xlabel = style.get("xlabel", defaults.get("xlabel"))
    ylabel = style.get("ylabel", defaults.get("ylabel"))
    title = style.get("title", defaults.get("title"))
    font_size = style.get("font_size")
    title_size = style.get("title_size", font_size)
    label_size = style.get("label_size", font_size)
    tick_size = style.get("tick_size", font_size)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size)
    if title:
        ax.set_title(title, fontsize=title_size)

    if tick_size:
        ax.tick_params(labelsize=tick_size)

def _apply_figsize(fig, style: dict):
    w = style.get("fig_width") or style.get("width")
    h = style.get("fig_height") or style.get("height")
    if w or h:
        cur_w, cur_h = fig.get_size_inches()
        fig.set_size_inches(float(w or cur_w), float(h or cur_h))

def _legend_kwargs(style: dict) -> dict:
    legend_title = style.get("legend_title")
    loc = style.get("legend_loc", "upper right")
    legend_fontsize = style.get("legend_font_size", style.get("font_size"))
    kwargs = {"loc": loc}
    if legend_title:
        kwargs["title"] = str(legend_title)
    if legend_fontsize:
        kwargs["fontsize"] = legend_fontsize
    return kwargs

def _get_ybreak_axes(y_break):
    if not y_break or len(y_break) != 2:
        return None, None
    try:
        # ordena de menor a mayor rango y coloca el mayor arriba
        (a_low, a_high), (b_low, b_high) = y_break
        segments = sorted([(a_low, a_high), (b_low, b_high)], key=lambda r: r[0])
        low_seg, high_seg = segments[0], segments[1]
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
        # el rango mayor arriba
        ax_top.set_ylim(high_seg[0], high_seg[1])
        ax_bot.set_ylim(low_seg[0], low_seg[1])
        # quitar spines para simular corte
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False)
        ax_bot.tick_params(labelbottom=True)
        d = .5
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)
        return fig, (ax_top, ax_bot)
    except Exception:
        return None, None


def plot_lines_vs_load(df: pd.DataFrame, out_path: Path, metric_base: str, metric_label: str, style: dict, ctx: dict, y_break=None, interpolate=False):
    mean_col = f"{metric_base}_mean"
    low_col  = f"{metric_base}_ci95_low"
    high_col = f"{metric_base}_ci95_high"

    fig, axes = _get_ybreak_axes(y_break)
    ax_list = axes if axes else [plt.subplots()[1]]
    if not axes:
        fig = ax_list[0].figure
    _apply_figsize(fig, style)
    _apply_figsize(fig, style)
    loads = sorted(df["load"].dropna().unique().tolist(), key=natural_load_key)
    x = list(range(len(loads)))
    load_suffix = style.get("load_suffix", None)
    disp_loads = [format_load_label(ld, load_suffix) for ld in loads]
    for ax in ax_list:
        ax.set_xticks(x)
        ax.set_xticklabels(disp_loads)

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

        y_s = clip_nonneg(pd.Series(y))
        lo_s = clip_nonneg(pd.Series(lo))
        hi_s = clip_nonneg(pd.Series(hi))

        # interp opcional si loads son numéricos
        numeric_x = [natural_load_key(ld) for ld in loads]
        can_interp = interpolate and len(loads) > 1 and all(isinstance(v, (int, float)) for v in numeric_x)

        for ax in ax_list:
            if can_interp:
                x_dense = np.linspace(min(numeric_x), max(numeric_x), 200)
                y_dense = np.interp(x_dense, numeric_x, y_s)
                ax.plot(x_dense, y_dense, label=ctrl)
                ax.scatter(numeric_x, y_s, s=16, label=None)
                if lo_s.notna().any() and hi_s.notna().any():
                    lo_dense = np.interp(x_dense, numeric_x, lo_s)
                    hi_dense = np.interp(x_dense, numeric_x, hi_s)
                    ax.fill_between(x_dense, lo_dense, hi_dense, alpha=0.15)
            else:
                ax.plot(x, y_s, marker="o", markersize=4, label=ctrl)

        mask = lo_s.notna() & hi_s.notna() & y_s.notna()
        if mask.any():
            for ax in ax_list:
                ax.fill_between(pd.Series(x)[mask], lo_s[mask], hi_s[mask], alpha=0.15)

    legend_title = style.get("legend_title")
    grid = bool(style.get("grid", False))
    ax_list[0].legend(**_legend_kwargs(style))

    defaults = {"xlabel": "Carga", "ylabel": metric_label, "title": f"{metric_label} vs carga"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    for i, ax in enumerate(ax_list):
        style_apply = dict(style2)
        # solo el eje superior mantiene título; el inferior no
        if i != 0:
            style_apply["title"] = ""
        # custom labels para cargas
        load_suffix_local = style_apply.pop("load_suffix", load_suffix)
        disp_loads_local = [format_load_label(ld, load_suffix_local) for ld in loads]
        _apply_axes_style(ax, style_apply, defaults)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticklabels(disp_loads_local)
    if len(ax_list) > 1:
        ax_list[0].set_xlabel("")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_ids_vs_load(df: pd.DataFrame, out_path: Path, metric_base: str, metric_label: str, ids: List[str],
                     id_labels: dict, style: dict, ctx: dict, y_break=None, interpolate: bool = False):
    mean_col = f"{metric_base}_mean"
    low_col  = f"{metric_base}_ci95_low"
    high_col = f"{metric_base}_ci95_high"

    fig, axes = _get_ybreak_axes(y_break)
    ax_list = axes if axes else [plt.subplots()[1]]
    if not axes:
        fig = ax_list[0].figure
    _apply_figsize(fig, style)
    loads = sorted(df["load"].dropna().unique().tolist(), key=natural_load_key)
    x = list(range(len(loads)))
    load_suffix = style.get("load_suffix", None)
    disp_loads = [format_load_label(ld, load_suffix) for ld in loads]
    for ax in ax_list:
        ax.set_xticks(x)
        ax.set_xticklabels(disp_loads)

    for idv in ids:
        d = df[df["id"] == idv].copy()
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

        y_s = clip_nonneg(pd.Series(y))
        lo_s = clip_nonneg(pd.Series(lo))
        hi_s = clip_nonneg(pd.Series(hi))

        numeric_x = [natural_load_key(ld) for ld in loads]
        can_interp = interpolate and len(loads) > 1 and all(isinstance(v, (int, float)) for v in numeric_x)

        label_id = display_id(id_labels, idv)
        for ax in ax_list:
            if can_interp:
                x_dense = np.linspace(min(numeric_x), max(numeric_x), 200)
                y_dense = np.interp(x_dense, numeric_x, y_s)
                ax.plot(x_dense, y_dense, label=label_id)
                ax.scatter(numeric_x, y_s, s=16, label=None)
                if lo_s.notna().any() and hi_s.notna().any():
                    lo_dense = np.interp(x_dense, numeric_x, lo_s)
                    hi_dense = np.interp(x_dense, numeric_x, hi_s)
                    ax.fill_between(x_dense, lo_dense, hi_dense, alpha=0.15)
            else:
                ax.plot(x, y_s, marker="o", markersize=4, label=label_id)
                if lo_s.notna().any() and hi_s.notna().any():
                    ax.fill_between(pd.Series(x), lo_s, hi_s, alpha=0.15)

    legend_title = style.get("legend_title")
    grid = bool(style.get("grid", False))
    ax_list[0].legend(**_legend_kwargs(style))

    defaults = {"xlabel": "Carga", "ylabel": metric_label, "title": style.get("title") or f"{metric_label} vs carga"}
    style2 = dict(style)
    for i, ax in enumerate(ax_list):
        style_apply = dict(style2)
        if i != 0:
            style_apply["title"] = ""
        _apply_axes_style(ax, style_apply, defaults)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.4)
    if len(ax_list) > 1:
        ax_list[0].set_xlabel("")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_grouped_bars(df: pd.DataFrame, out_path: Path, metric_base: str, metric_label: str, style: dict, ctx: dict, loads_order: Optional[List[str]] = None, y_break=None):
    mean_col = f"{metric_base}_mean"
    low_col  = f"{metric_base}_ci95_low"
    high_col = f"{metric_base}_ci95_high"

    loads = loads_order or sorted(df["load"].dropna().unique().tolist(), key=natural_load_key)
    controllers = sorted(df["controller"].dropna().unique().tolist())

    width = 0.8
    n_ctrl = max(1, len(controllers))
    bar_w = width / n_ctrl
    x = list(range(len(loads)))

    fig, axes = _get_ybreak_axes(y_break)
    ax_list = axes if axes else [plt.subplots()[1]]
    if not axes:
        fig = ax_list[0].figure
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

        ys_s = clip_nonneg(pd.Series(ys))
        yerr_low_s = pd.Series(yerr_low)
        yerr_high_s = pd.Series(yerr_high)
        # recalc err si mean se clippea a 0
        yerr_low_s = yerr_low_s.where(ys_s > 0, 0.0)
        yerr_high_s = yerr_high_s.where(ys_s.notna(), 0.0)

        offsets = [xi - width/2 + (j + 0.5)*bar_w for xi in x]
        for ax in ax_list:
            ax.bar(offsets, ys_s, bar_w, yerr=[yerr_low_s, yerr_high_s], capsize=3, label=ctrl)

    load_suffix = style.get("load_suffix", None)
    disp_loads = [format_load_label(ld, load_suffix) for ld in loads]
    for ax in ax_list:
        ax.set_xticks(x)
        ax.set_xticklabels(disp_loads)

    legend_title = style.get("legend_title")
    grid = bool(style.get("grid", False))
    ax_list[0].legend(**_legend_kwargs(style))

    defaults = {"xlabel": "Carga", "ylabel": metric_label, "title": f"{metric_label} vs carga"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    for i, ax in enumerate(ax_list):
        style_apply = dict(style2)
        if i != 0:
            style_apply["title"] = ""
        _apply_axes_style(ax, style_apply, defaults)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.4)
    if len(ax_list) > 1:
        ax_list[0].set_xlabel("")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_multi_controllers(ts_df: pd.DataFrame, out_path: Path, metric: str, metric_label: str,
                                      min_reps_per_t: int, crop_to_common_time: bool,
                                      style: dict, ctx: dict, show_ci: bool = True, smooth_cfg: Optional[dict] = None):
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
    _apply_figsize(fig, style)
    for ctrl in sorted(df["controller"].dropna().unique().tolist()):
        d = df[df["controller"] == ctrl].sort_values("t_rel_s")
        d[mean_col] = smooth_series(d[mean_col], smooth_cfg)
        if show_ci and low_col in d.columns and high_col in d.columns:
            d[low_col] = smooth_series(d[low_col], smooth_cfg)
            d[high_col] = smooth_series(d[high_col], smooth_cfg)
        ax.plot(d["t_rel_s"], d[mean_col], label=ctrl)
        if show_ci:
            dd = d.dropna(subset=[low_col, high_col, mean_col])
            if not dd.empty:
                ax.fill_between(dd["t_rel_s"], dd[low_col], dd[high_col], alpha=0.15)

    legend_title = style.get("legend_title")
    grid = bool(style.get("grid", False))
    ax.legend(**_legend_kwargs(style))

    defaults = {"xlabel": "Tiempo relativo (s)", "ylabel": metric_label, "title": f"{metric_label} vs tiempo"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    _apply_axes_style(ax, style2, defaults)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_multi_loads(ts_df: pd.DataFrame, out_path: Path, metric: str, metric_label: str,
                                min_reps_per_t: int, crop_to_common_time: bool,
                                style: dict, ctx: dict, show_ci: bool = True, smooth_cfg: Optional[dict] = None):
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
    _apply_figsize(fig, style)
    loads = sorted(df["load"].dropna().unique().tolist(), key=natural_load_key)
    for ld in loads:
        d = df[df["load"] == ld].sort_values("t_rel_s")
        d[mean_col] = smooth_series(d[mean_col], smooth_cfg)
        if show_ci and low_col in d.columns and high_col in d.columns:
            d[low_col] = smooth_series(d[low_col], smooth_cfg)
            d[high_col] = smooth_series(d[high_col], smooth_cfg)
        ax.plot(d["t_rel_s"], d[mean_col], label=str(ld))
        if show_ci:
            dd = d.dropna(subset=[low_col, high_col, mean_col])
            if not dd.empty:
                ax.fill_between(dd["t_rel_s"], dd[low_col], dd[high_col], alpha=0.15)

    legend_title = style.get("legend_title", "Carga")
    grid = bool(style.get("grid", False))
    style_legend = dict(style)
    style_legend.setdefault("legend_title", legend_title)
    ax.legend(**_legend_kwargs(style_legend))

    defaults = {"xlabel": "Tiempo relativo (s)", "ylabel": metric_label, "title": f"{metric_label} vs tiempo"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    _apply_axes_style(ax, style2, defaults)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_compare_ids(ts_df: pd.DataFrame, out_path: Path, metric: str, metric_label: str,
                                ids: List[str], controllers: List[str],
                                id_labels: dict, style: dict, ctx: dict,
                                min_reps_per_t: int, crop_to_common_time: bool,
                                show_ci: bool = True, smooth_cfg: Optional[dict] = None):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"
    count_col = f"{metric}_count"

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

    if crop_to_common_time:
        tmax_list = [int(s[2]["t_rel_s"].max()) for s in series if not s[2].empty]
        if tmax_list:
            common_tmax = min(tmax_list)
            series = [(c, i, d[d["t_rel_s"] <= common_tmax].copy()) for (c, i, d) in series]

    fig, ax = plt.subplots()
    _apply_figsize(fig, style)
    multi_ctrl = len({c for c, _, _ in series}) > 1

    for ctrl, idv, d in series:
        label_id = display_id(id_labels, idv)
        label = f"{label_id} ({ctrl})" if multi_ctrl else label_id

        d = d.sort_values("t_rel_s")
        d[mean_col] = smooth_series(d[mean_col], smooth_cfg)
        if show_ci and low_col in d.columns and high_col in d.columns:
            d[low_col] = smooth_series(d[low_col], smooth_cfg)
            d[high_col] = smooth_series(d[high_col], smooth_cfg)

        ax.plot(d["t_rel_s"], d[mean_col], label=label)
        if show_ci and low_col in d.columns and high_col in d.columns:
            dd = d.dropna(subset=[low_col, high_col, mean_col])
            if not dd.empty:
                ax.fill_between(dd["t_rel_s"], dd[low_col], dd[high_col], alpha=0.15)

    legend_title = style.get("legend_title")
    grid = bool(style.get("grid", False))
    ax.legend(**_legend_kwargs(style))

    defaults = {"xlabel": "Tiempo relativo (s)", "ylabel": metric_label, "title": f"{metric_label} vs tiempo"}
    style2 = dict(style)
    if isinstance(style2.get("title"), str):
        style2["title"] = fmt_template(style2["title"], **ctx)
    _apply_axes_style(ax, style2, defaults)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Faceted compare (una curva por panel, apilados, sin leyenda)
def plot_timeseries_compare_ids_faceted(ts_df: pd.DataFrame, out_path: Path, metric: str, metric_label: str,
                                        ids: List[str], controllers: List[str],
                                        id_labels: dict, style: dict, ctx: dict,
                                        min_reps_per_t: int, crop_to_common_time: bool,
                                        show_ci: bool = True, smooth_cfg: Optional[dict] = None):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"
    count_col = f"{metric}_count"

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

    if crop_to_common_time:
        tmax_list = [int(s[2]["t_rel_s"].max()) for s in series if not s[2].empty]
        if tmax_list:
            common_tmax = min(tmax_list)
            series = [(c, i, d[d["t_rel_s"] <= common_tmax].copy()) for (c, i, d) in series]

    y_max = 0.0
    for _, _, d in series:
        candidates = [d[mean_col].max(skipna=True)]
        if show_ci and high_col in d.columns:
            candidates.append(d[high_col].max(skipna=True))
        cand = pd.to_numeric(pd.Series(candidates), errors="coerce").dropna()
        if not cand.empty:
            y_max = max(y_max, float(cand.max()))
    y_max = y_max * 1.05 if y_max > 0 else None

    n = len(series)
    fig_height = max(2.5, 2.3 * n)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8, fig_height))
    _apply_figsize(fig, style)
    if n == 1:
        axes = [axes]

    grid = bool(style.get("grid", False))
    style_no_legend = dict(style)
    style_no_legend.pop("legend_title", None)
    style_no_legend.pop("legend_loc", None)

    for ax, (ctrl, idv, d) in zip(axes, series):
        d = d.sort_values("t_rel_s")
        d[mean_col] = smooth_series(d[mean_col], smooth_cfg)
        if show_ci and low_col in d.columns and high_col in d.columns:
            d[low_col] = smooth_series(d[low_col], smooth_cfg)
            d[high_col] = smooth_series(d[high_col], smooth_cfg)

        ax.plot(d["t_rel_s"], d[mean_col], label=None, color="#1f77b4")
        if show_ci and low_col in d.columns and high_col in d.columns:
            dd = d.dropna(subset=[low_col, high_col, mean_col])
            if not dd.empty:
                ax.fill_between(dd["t_rel_s"], dd[low_col], dd[high_col], alpha=0.15, color="#1f77b4")

        if y_max:
            ax.set_ylim(0, y_max)

        label_id = display_id(id_labels, idv)
        # Título = etiqueta amigable del id (id_labels)
        title_txt = str(label_id)
        local_style = dict(style_no_legend)
        local_style["title"] = title_txt
        _apply_axes_style(ax, local_style, {"xlabel": None, "ylabel": metric_label, "title": title_txt})
        if grid:
            ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel(style.get("xlabel", "Tiempo relativo (s)"))

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
    stats_cfg = cfg.setdefault("stats", {})
    stats_cfg.setdefault("confidence", {"enable": True, "level": 0.95, "method": "t"})
    stats_cfg.setdefault("outliers", {"enable": False, "metrics": ["jitter_mean_ms"], "zscore_threshold": 3.0})
    stats_cfg.setdefault("time_window", {"enable": False, "start_s": 0, "end_s": None, "require_full_window": True})

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
    p_time_report = out_dir / cfg["outputs"]["csv"].get("time_coverage", "time_coverage.csv")

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
        time_report_df = read_csv_maybe_empty(p_time_report)
        removed_outliers_df = pd.DataFrame()
        time_outliers = []
    else:
        missing: List[dict] = []
        rep_flow_rows: List[dict] = []
        rep_proto_acc: Dict[Tuple[str, str, str, str], dict] = {}
        time_report_rows: List[dict] = []
        time_outliers: List[dict] = []

        ts_flow_rows: List[dict] = []
        ts_proto_rows_raw: List[dict] = []
        ts_flow_meta: List[dict] = []
        ts_proto_meta: List[dict] = []

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
                        flow_summary, proto_accum, df_ts, df_ts_proto, time_meta = compute_file_rep_summaries(df_norm, cfg, flow_id, proto)
                    except Exception as e:
                        missing.append({"controller": controller, "load": load, "rep": rep, "file": str(csv_path), "reason": f"compute_error:{e}"})
                        continue
                    time_meta = time_meta or {}
                    time_meta.update({"controller": controller, "load": load, "rep": rep, "file": str(csv_path)})
                    time_report_rows.append(time_meta)

                    tw_cfg = (cfg.get("stats", {}) or {}).get("time_window", {}) or {}
                    tw_enable = bool(tw_cfg.get("enable", False))
                    if tw_enable and not time_meta.get("window_ok", True):
                        time_outliers.append({
                            "controller": controller,
                            "load": load,
                            "rep": rep,
                            "id": flow_id,
                            "metric": "time_window",
                            "value": time_meta.get("t_rel_max"),
                            "reason": "insufficient_time_window",
                            "required_end_s": tw_cfg.get("end_s", tw_cfg.get("max_s")),
                        })
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
                    if not df_ts.empty:
                        ts_flow_meta.append({
                            "controller": controller, "load": load, "rep": rep, "id": flow_id,
                            "t_max": int(df_ts["t_rel_s"].max())
                        })

                    # rep protocol raw rows
                    for _, r in df_ts_proto.iterrows():
                        ts_proto_rows_raw.append({
                            "controller": controller, "load": load, "rep": rep,
                            "protocol": proto,
                            "t_sec": float(r["t_sec"]) if pd.notna(r["t_sec"]) else None,
                            "throughput_bytes": float(r["throughput_bytes"]) if pd.notna(r["throughput_bytes"]) else 0.0,
                        })
                    if not df_ts_proto.empty:
                        ts_proto_meta.append({
                            "controller": controller, "load": load, "rep": rep, "id": proto,
                            "t_max": float(df_ts_proto["t_sec"].max()) if df_ts_proto["t_sec"].notna().any() else 0.0,
                            "t_min": float(df_ts_proto["t_sec"].min()) if df_ts_proto["t_sec"].notna().any() else 0.0,
                        })

        rep_flow_df = pd.DataFrame(rep_flow_rows)
        rep_proto_df = pd.DataFrame(list(rep_proto_acc.values()))
        miss_df = pd.DataFrame(missing)
        time_report_df = pd.DataFrame(time_report_rows)

        # outlier filter por repetición
        rep_flow_df, removed_outliers_df = filter_outliers(rep_flow_df, cfg)
        if time_outliers:
            df_time_out = pd.DataFrame(time_outliers)
            removed_outliers_df = pd.concat([removed_outliers_df, df_time_out], ignore_index=True) if not removed_outliers_df.empty else df_time_out

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
                        lo, hi = clamp_ci(lo, hi)
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

        crop_common_ts = bool(cfg["timeseries"].get("crop_to_common_time", False))

        # timeseries mean/CI over reps (flow)
        ts_flow_mean_df = pd.DataFrame()
        if ts_flow_rows:
            df_ts_rep = pd.DataFrame(ts_flow_rows)
            if crop_common_ts and ts_flow_meta:
                meta_df = pd.DataFrame(ts_flow_meta)
                mins = (
                    meta_df.groupby(["controller", "load", "id"])["t_max"]
                    .min()
                    .reset_index()
                    .rename(columns={"t_max": "t_max_min"})
                )
                df_ts_rep = df_ts_rep.merge(mins, on=["controller", "load", "id"], how="left")
                df_ts_rep = df_ts_rep[df_ts_rep["t_rel_s"] <= df_ts_rep["t_max_min"]].drop(columns=["t_max_min"])
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
                    lo, hi = clamp_ci(lo, hi)
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
                if crop_common_ts and ts_proto_meta:
                    meta_df = pd.DataFrame(ts_proto_meta)
                    # meta t_max está en segundos absolutos; convertir a t_rel_max por rep usando t_max - t_min
                    meta_df["t_max_rel"] = meta_df["t_max"] - meta_df["t_min"]
                    mins = (
                        meta_df.groupby(["controller", "load", "id"])["t_max_rel"]
                        .min()
                        .reset_index()
                        .rename(columns={"t_max_rel": "t_max_min"})
                    )
                    df_rep = df_rep.merge(mins, on=["controller", "load", "id"], how="left")
                    df_rep = df_rep[df_rep["t_rel_s"] <= df_rep["t_max_min"]].drop(columns=["t_max_min"])
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
                        lo, hi = clamp_ci(lo, hi)
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
        time_report_df.to_csv(p_time_report, index=False)
        outliers_path = None
        if "outputs" in cfg and "csv" in cfg["outputs"]:
            outliers_rel = cfg["outputs"]["csv"].get("outliers", None)
            if outliers_rel:
                outliers_path = (out_dir / outliers_rel)
                if not removed_outliers_df.empty:
                    outliers_path.parent.mkdir(parents=True, exist_ok=True)
                    removed_outliers_df.to_csv(outliers_path, index=False)

        print(f"OK: {p_rep_flow}")
        print(f"OK: {p_rep_proto}")
        print(f"OK: {p_load_flow}")
        print(f"OK: {p_load_proto}")
        print(f"OK: {p_ts_flow}")
        print(f"OK: {p_ts_proto}")
        print(f"OK: {p_missing}")
        print(f"OK: {p_time_report}")
        if outliers_path and not removed_outliers_df.empty:
            print(f"OK: outliers -> {outliers_path}")

    # plots
    plots = cfg.get("plots", {}) or {}
    metric_labels_cfg = cfg.get("metric_labels", {}) or {}

    def resolve_style(pcfg: dict) -> dict:
        style_raw = (pcfg.get("labels") if isinstance(pcfg, dict) else None) or (pcfg.get("style") if isinstance(pcfg, dict) else {}) or {}
        style = dict(style_raw) if isinstance(style_raw, dict) else {}
        if "legend" in style and "legend_title" not in style:
            style["legend_title"] = style.get("legend")
        if "grid" not in style and isinstance(pcfg, dict) and "grid" in pcfg:
            style["grid"] = pcfg.get("grid")
        # Copia campos de estilo declarados al nivel superior del bloque (comodidad)
        passthrough_keys = [
            "legend_loc", "legend_font_size", "font_size",
            "title_size", "label_size", "tick_size", "load_suffix",
            "fig_width", "fig_height", "width", "height",
        ]
        for k in passthrough_keys:
            if k not in style and isinstance(pcfg, dict) and k in pcfg:
                style[k] = pcfg.get(k)
        return style

    def pick_ids_from_labels(id_labels: dict, ids_sel, universe: List[str]) -> List[str]:
        # Si id_labels se define, solo usa esas claves como selección implícita.
        if isinstance(id_labels, dict) and id_labels:
            return [str(k) for k in id_labels.keys() if str(k) in universe]
        if ids_sel is None or ids_sel == "all":
            return universe
        if isinstance(ids_sel, list):
            return [str(x) for x in ids_sel]
        return [str(ids_sel)]

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

    def ybreak_for_id(yb_cfg, idv):
        if yb_cfg is None:
            return None
        # si viene como lista de 2 rangos, aplica a todos
        if isinstance(yb_cfg, list) and len(yb_cfg) == 2 and all(isinstance(r, list) for r in yb_cfg):
            return yb_cfg
        if isinstance(yb_cfg, dict):
            if idv in yb_cfg:
                return yb_cfg[idv]
            return yb_cfg.get("default")
        return None

    # delay vs load
    if plots.get("delay_vs_load", {}).get("enable", False):
        pcfg = plots["delay_vs_load"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", "delay_mean_ms"))
        style = resolve_style(pcfg)
        metric_label = get_metric_label(metric_labels_cfg, metric)
        ids_sel = pcfg.get("ids", "all")
        id_labels = pcfg.get("id_labels", {}) or {}
        yb_cfg = pcfg.get("y_break", None)
        interp = bool(pcfg.get("interpolate", False))
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = pick_ids_from_labels(id_labels, ids_sel, sorted(dfL["id"].dropna().unique().tolist()))
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                if f"{metric}_mean" not in d.columns:
                    continue
                outp = plots_dir / f"delay_vs_load_{group}_{idv}.png"
                label_id = display_id(id_labels, idv)
                y_break = ybreak_for_id(yb_cfg, idv)
                plot_lines_vs_load(
                    d, outp, metric, metric_label, style,
                    {"id": label_id, "id_label": label_id, "group": group, "metric": metric},
                    y_break=y_break,
                    interpolate=interp,
                )
                print(f"OK plot: {outp}")

    # jitter vs load
    if plots.get("jitter_vs_load", {}).get("enable", False):
        pcfg = plots["jitter_vs_load"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", "jitter_mean_ms"))
        style = resolve_style(pcfg)
        metric_label = get_metric_label(metric_labels_cfg, metric)
        ids_sel = pcfg.get("ids", "all")
        id_labels = pcfg.get("id_labels", {}) or {}
        yb_cfg = pcfg.get("y_break", None)
        interp = bool(pcfg.get("interpolate", False))
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = pick_ids_from_labels(id_labels, ids_sel, sorted(dfL["id"].dropna().unique().tolist()))
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                ycol = f"{metric}_mean"
                if ycol not in d.columns:
                    continue
                if (cfg.get("jitter", {}) or {}).get("plot_only_if_present", True) and d[ycol].dropna().empty:
                    continue
                outp = plots_dir / f"jitter_vs_load_{group}_{idv}.png"
                label_id = display_id(id_labels, idv)
                y_break = ybreak_for_id(yb_cfg, idv)
                plot_lines_vs_load(
                    d, outp, metric, metric_label, style,
                    {"id": label_id, "id_label": label_id, "group": group, "metric": metric},
                    y_break=y_break,
                    interpolate=interp,
                )
                print(f"OK plot: {outp}")

    # loss bars
    if plots.get("loss_bars", {}).get("enable", False):
        pcfg = plots["loss_bars"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", "loss_pct"))
        style = resolve_style(pcfg)
        metric_label = get_metric_label(metric_labels_cfg, metric)
        loads_sel = pcfg.get("loads", "all")
        ids_sel = pcfg.get("ids", "all")
        id_labels = pcfg.get("id_labels", {}) or {}
        yb_cfg = pcfg.get("y_break", None)
        interp = bool(pcfg.get("interpolate", False))
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = pick_ids_from_labels(id_labels, ids_sel, sorted(dfL["id"].dropna().unique().tolist()))
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                if f"{metric}_mean" not in d.columns:
                    continue
                available_loads = sorted(d["load"].dropna().unique().tolist(), key=natural_load_key)
                loads_list = select_list(loads_sel, available_loads)
                d = d[d["load"].isin(loads_list)]
                if d.empty:
                    continue
                outp = plots_dir / f"loss_lines_{group}_{idv}.png"
                label_id = display_id(id_labels, idv)
                y_break = ybreak_for_id(yb_cfg, idv)
                plot_lines_vs_load(
                    d,
                    outp,
                    metric,
                    metric_label,
                    style,
                    {"id": label_id, "id_label": label_id, "group": group, "metric": metric},
                    y_break=y_break,
                    interpolate=interp,
                )
                print(f"OK plot: {outp}")

    # throughput bars
    if plots.get("throughput_bars", {}).get("enable", False):
        pcfg = plots["throughput_bars"]
        group = str(pcfg.get("group", "flow"))
        metric = str(pcfg.get("metric", f"throughput_total_mean_{thr_unit}"))
        style = resolve_style(pcfg)
        metric_label = get_metric_label(metric_labels_cfg, metric)
        loads_sel = pcfg.get("loads", "all")
        ids_sel = pcfg.get("ids", "all")
        id_labels = pcfg.get("id_labels", {}) or {}
        yb_cfg = pcfg.get("y_break", None)
        interp = bool(pcfg.get("interpolate", False))
        dfL = pick_load_df(group)
        if not dfL.empty:
            ids = pick_ids_from_labels(id_labels, ids_sel, sorted(dfL["id"].dropna().unique().tolist()))
            for idv in ids:
                d = dfL[dfL["id"] == idv].copy()
                if f"{metric}_mean" not in d.columns:
                    continue
                available_loads = sorted(d["load"].dropna().unique().tolist(), key=natural_load_key)
                loads_list = select_list(loads_sel, available_loads)
                d = d[d["load"].isin(loads_list)]
                if d.empty:
                    continue
                outp = plots_dir / f"throughput_lines_{group}_{idv}.png"
                label_id = display_id(id_labels, idv)
                y_break = ybreak_for_id(yb_cfg, idv)
                plot_lines_vs_load(
                    d,
                    outp,
                    metric,
                    metric_label,
                    style,
                    {"id": label_id, "id_label": label_id, "group": group, "metric": metric},
                    y_break=y_break,
                    interpolate=interp,
                )
                print(f"OK plot: {outp}")

    # throughput timeseries for one load (curves = controllers)
    if plots.get("throughput_timeseries", {}).get("enable", False):
        pcfg = plots["throughput_timeseries"]
        group = str(pcfg.get("group", "flow"))
        style = resolve_style(pcfg)
        sel = pcfg.get("select", {}) or {}
        load_sel = sel.get("load")
        id_sel = sel.get("id")
        controller_sel = sel.get("controller", "all")
        metric_label = get_metric_label(metric_labels_cfg, "throughput_Mbps")
        show_ci = bool(pcfg.get("show_ci", True))
        smooth_cfg = pcfg.get("smooth", {})
        id_labels = pcfg.get("id_labels", {}) or {}

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
                    label_id = display_id(id_labels, one_id)
                    plot_timeseries_multi_controllers(
                        d0, outp, "throughput_Mbps", metric_label,
                        min_reps, crop_common,
                        style, {"id": label_id, "id_label": label_id, "group": group, "load": load_sel, "metric": "throughput_Mbps"},
                        show_ci=show_ci, smooth_cfg=smooth_cfg,
                    )
                    print(f"OK plot: {outp}")

    # Comparación de flujos por controlador: líneas vs carga, varios ids en la misma gráfica
    if plots.get("compare_flows_by_controller", {}).get("enable", False):
        pcfg = plots["compare_flows_by_controller"]
        group = str(pcfg.get("group", "flow"))
        ids_sel = pcfg.get("ids", "all")
        id_labels = pcfg.get("id_labels", {}) or {}
        metrics_sel = pcfg.get("metrics", [])
        controllers_sel = pcfg.get("controllers", "all")
        loads_sel = pcfg.get("loads", "all")
        yb_cfg = pcfg.get("y_break", None)
        interp = bool(pcfg.get("interpolate", False))
        style = resolve_style(pcfg)

        dfL = pick_load_df(group)
        if not dfL.empty:
            controllers = sorted(dfL["controller"].dropna().unique().tolist())
            if controllers_sel != "all":
                if isinstance(controllers_sel, list):
                    controllers = [c for c in controllers if c in controllers_sel]
                else:
                    controllers = [c for c in controllers if c == controllers_sel]
            ids_all = sorted(dfL["id"].dropna().unique().tolist())
            ids = pick_ids_from_labels(id_labels, ids_sel, ids_all)
            loads_all = sorted(dfL["load"].dropna().unique().tolist(), key=natural_load_key)
            loads = select_list(loads_sel, loads_all)

            for ctrl in controllers:
                d_ctrl = dfL[(dfL["controller"] == ctrl) & (dfL["load"].isin(loads))].copy()
                if d_ctrl.empty:
                    continue
                for metric in metrics_sel:
                    metric_label = get_metric_label(metric_labels_cfg, metric)
                    y_break = yb_cfg.get(metric) if isinstance(yb_cfg, dict) else yb_cfg
                    outp = plots_dir / f"compare_{metric}_{ctrl}.png"
                    style_cmp = dict(style)
                    style_cmp.setdefault("ylabel", metric_label)
                    plot_ids_vs_load(
                        d_ctrl,
                        outp,
                        metric,
                        metric_label,
                        ids,
                        id_labels,
                        style_cmp,
                        {"controller": ctrl, "metric": metric},
                        y_break=y_break,
                        interpolate=interp,
                    )
                    print(f"OK plot: {outp}")

    # NEW: throughput timeseries by load (curves = loads)
    if plots.get("throughput_timeseries_loads", {}).get("enable", False):
        pcfg = plots["throughput_timeseries_loads"]
        group = str(pcfg.get("group", "flow"))
        style = resolve_style(pcfg)
        sel = pcfg.get("select", {}) or {}
        id_sel = sel.get("id", "all")
        controller_sel = sel.get("controller", "all")
        loads_sel = sel.get("loads", "all")
        metric_label = get_metric_label(metric_labels_cfg, "throughput_Mbps")
        show_ci = bool(pcfg.get("show_ci", True))
        smooth_cfg = pcfg.get("smooth", {})
        id_labels = pcfg.get("id_labels", {}) or {}

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
                    label_id = display_id(id_labels, one_id)
                    plot_timeseries_multi_loads(
                        d0, outp, "throughput_Mbps", metric_label,
                        min_reps, crop_common,
                        style, {"id": label_id, "id_label": label_id, "group": group, "controller": one_ctrl, "metric": "throughput_Mbps"},
                        show_ci=show_ci, smooth_cfg=smooth_cfg,
                    )
                    print(f"OK plot: {outp}")

    # NEW: throughput timeseries compare (misma gráfica, varias ids) por carga
    if plots.get("throughput_timeseries_compare", {}).get("enable", False):
        pcfg = plots["throughput_timeseries_compare"]
        group = str(pcfg.get("group", "flow"))
        style = resolve_style(pcfg)
        sel = pcfg.get("select", {}) or {}
        ids_sel = sel.get("id", "all")
        controller_sel = sel.get("controller", "all")
        loads_sel = sel.get("loads", "all")
        metric_label = get_metric_label(metric_labels_cfg, "throughput_Mbps")
        show_ci = bool(pcfg.get("show_ci", True))
        smooth_cfg = pcfg.get("smooth", {})
        id_labels = pcfg.get("id_labels", {}) or {}
        faceted = bool(pcfg.get("faceted", False))

        dfT = pick_ts_df(group)
        if not dfT.empty:
            ids_all = sorted(dfT["id"].dropna().unique().tolist())
            ctrls_all = sorted(dfT["controller"].dropna().unique().tolist())
            loads_all = sorted(dfT["load"].dropna().unique().tolist(), key=natural_load_key)

            ids = select_list(ids_sel, ids_all)
            ctrls = select_list(controller_sel, ctrls_all)
            loads = select_list(loads_sel, loads_all)

            min_reps = int(cfg["timeseries"].get("min_reps_per_t", 1))
            crop_common = bool(cfg["timeseries"].get("crop_to_common_time", False))

            for load_sel in loads:
                df_load = dfT[dfT["load"] == load_sel].copy()
                if df_load.empty:
                    continue

                ids_use = [i for i in ids if i in df_load["id"].unique()]
                ctrls_use = [c for c in ctrls if c in df_load["controller"].unique()]
                if not ids_use or not ctrls_use:
                    continue

                ctrl_slug = slugify("-".join(ctrls_use))
                fname = f"ts_throughput_compare_{group}_{load_sel}_{ctrl_slug}"
                if faceted:
                    fname += "_faceted"
                fname += ".png"
                outp = plots_dir / fname
                ctx = {"load": load_sel, "controller": ",".join(ctrls_use), "ids": ", ".join(ids_use), "group": group, "metric": "throughput_Mbps"}
                if faceted:
                    plot_timeseries_compare_ids_faceted(
                        df_load, outp, "throughput_Mbps", metric_label,
                        ids_use, ctrls_use, id_labels, style, ctx,
                        min_reps_per_t=min_reps, crop_to_common_time=crop_common,
                        show_ci=show_ci, smooth_cfg=smooth_cfg,
                    )
                else:
                    plot_timeseries_compare_ids(
                        df_load, outp, "throughput_Mbps", metric_label,
                        ids_use, ctrls_use, id_labels, style, ctx,
                        min_reps_per_t=min_reps, crop_to_common_time=crop_common,
                        show_ci=show_ci, smooth_cfg=smooth_cfg,
                    )
                print(f"OK plot: {outp}")
        else:
            print("WARNING: no hay datos de timeseries para throughput_timeseries_compare")

if __name__ == "__main__":
    main()
