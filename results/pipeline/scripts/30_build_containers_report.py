#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -------------------------
# Helpers
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
    # "1a" -> 1.0, "1.2a" -> 1.2, fallback: string
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
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

def bytes_to_mib(x: float) -> float:
    return x / (1024.0 * 1024.0)

def bytes_to_kib(x: float) -> float:
    return x / 1024.0

TRANSFORMS = {
    "bytes_to_mib": bytes_to_mib,
    "bytes_to_kib": bytes_to_kib,
}


def metric_display_name(metrics_cfg: dict, mname: str) -> str:
    m = metrics_cfg.get(mname, {}) if isinstance(metrics_cfg, dict) else {}
    base = m.get("label") or mname
    unit = m.get("unit")
    return f"{base} ({unit})" if unit else base

def apply_figsize(fig, labels_cfg: Optional[dict]):
    if not labels_cfg:
        return
    w = labels_cfg.get("fig_width") or labels_cfg.get("width")
    h = labels_cfg.get("fig_height") or labels_cfg.get("height")
    if w or h:
        cur_w, cur_h = fig.get_size_inches()
        fig.set_size_inches(float(w or cur_w), float(h or cur_h))


def smooth_series(s: pd.Series, smooth_cfg: Optional[dict]) -> pd.Series:
    if not smooth_cfg or not smooth_cfg.get("enable", False):
        return s
    try:
        w = int(smooth_cfg.get("window_points", 1))
    except Exception:
        w = 1
    if w <= 1:
        return s
    # rolling mean centrada para suavizar curva visual
    return s.rolling(window=w, min_periods=1, center=True).mean()

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def delta_last_first(s: pd.Series) -> Optional[float]:
    v = to_numeric_series(s).dropna()
    if len(v) < 2:
        return None
    d = float(v.iloc[-1]) - float(v.iloc[0])
    # si hubo reset del contador, lo dejamos como None
    return d if d >= 0 else None

def t_critical(level: float, df: int) -> float:
    """
    t crítico para IC bilateral (level=0.95 -> 0.975).
    Requiere scipy. Si no está, cae a aproximación normal.
    """
    alpha = 1.0 - level
    p = 1.0 - alpha / 2.0
    try:
        from scipy.stats import t
        return float(t.ppf(p, df))
    except Exception:
        # fallback normal: razonable si n grande
        from math import erf, sqrt
        # inv cdf normal aproximado (Acklam simplificado) no lo metemos,
        # usamos 1.96 para 95% como fallback común
        if abs(level - 0.95) < 1e-6:
            return 1.96
        # si cambia level, usa 1.96 igual (o ajusta instalando scipy)
        return 1.96

def compute_ci(mean: float, std: float, n: int, level: float, method: str) -> Tuple[Optional[float], Optional[float]]:
    if n < 2 or mean is None or std is None or pd.isna(mean) or pd.isna(std):
        return (None, None)
    se = std / math.sqrt(n)
    if method == "t":
        crit = t_critical(level, n - 1)
    else:
        crit = 1.96 if abs(level - 0.95) < 1e-6 else 1.96
    return (mean - crit * se, mean + crit * se)

def clamp_ci(lo: Optional[float], hi: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if lo is not None and not pd.isna(lo) and lo < 0:
        lo = 0.0
    if hi is not None and not pd.isna(hi) and hi < 0:
        hi = 0.0
    return lo, hi

def resample_df(
    df: pd.DataFrame,
    resample_s: int,
    gauge_cols: List[str],
    counter_cols: List[str],
    method: str = "sample_hold",
    interp_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    df trae 't_rel_s' numérico.
    Retorna dataframe resampleado a grid regular (1s, 2s, etc) con:
      - gauges: mean
      - counters: last (y forward-fill)
    """
    # índice de timedelta para resample robusto
    idx = pd.to_timedelta(df["t_rel_s"], unit="s")
    df2 = df.copy()
    df2.index = idx

    freq = f"{int(resample_s)}S"

    out_parts = []
    if gauge_cols:
        method = (method or "sample_hold").lower()
        if method == "interpolate":
            g = df2[gauge_cols].resample(freq).mean()
            # interpolación lineal para rellenar huecos
            interp_method = (interp_cfg or {}).get("method", "linear")
            g = g.interpolate(method=interp_method, limit_direction="both")
            g = g.ffill().bfill()
            out_parts.append(g)
        else:
            # Sample-and-hold: toma el último valor de cada bin y rellena hacia adelante
            g = df2[gauge_cols].resample(freq).last().ffill()
            out_parts.append(g)
    if counter_cols:
        c = df2[counter_cols].resample(freq).last().ffill()
        out_parts.append(c)

    if not out_parts:
        return pd.DataFrame()

    out = pd.concat(out_parts, axis=1)
    # agrega t_rel_s como entero (segundos)
    out["t_rel_s"] = (out.index.total_seconds()).astype(int)
    out = out.reset_index(drop=True)
    return out


# -------------------------
# Warmup detection
# -------------------------

def detect_warmup_auto(cpu_series: pd.Series, t_series: pd.Series, cfg: dict) -> int:
    """
    cpu_series: ya resampleado (1s), con NaNs posibles
    t_series: segundos relativos enteros
    Devuelve warmup_start_s (int)
    """
    th = float(cfg["cpu_idle_th"])
    smooth_window_s = int(cfg["smooth_window_s"])
    sustain_s = int(cfg["sustain_s"])
    max_search_s = int(cfg["max_search_s"])

    cpu = to_numeric_series(cpu_series).fillna(0.0)

    # suavizado (rolling mean)
    win = max(1, smooth_window_s)
    cpu_smooth = cpu.rolling(window=win, min_periods=1).mean()

    active = (cpu_smooth > th).astype(int)
    sustain = max(1, sustain_s)
    # suma móvil: si suma == sustain => hubo sustain consecutivo
    run = active.rolling(window=sustain, min_periods=sustain).sum()

    # restringe búsqueda
    mask = (t_series <= max_search_s)
    candidates = run[mask & run.notna() & (run >= sustain)]
    if candidates.empty:
        return None  # caller aplica fallback

    # primer punto donde se cumple, pero el inicio real es sustain-1 antes
    first_idx = int(candidates.index[0])
    warmup_start = max(0, int(t_series.iloc[first_idx]) - (sustain - 1))
    return warmup_start


# -------------------------
# Metrics compute
# -------------------------

def compute_rep_summary(
    df_raw: pd.DataFrame,
    resample_s: int,
    resample_method: str,
    resample_interp_cfg: dict,
    warmup_cfg: dict,
    metrics_cfg: dict,
    required_cols: List[str],
    crop_t_rel_max: Optional[int] = None,
) -> Tuple[Dict[str, Optional[float]], Dict[str, object], pd.DataFrame]:
    """
    Retorna:
      - metrics_out: valores resumen (post-warmup si aplica)
      - meta: warmup_s, duration_used_s, samples_used, ...
      - df_resampled: df resampleado (para series de tiempo)
    """
    df = ensure_columns(df_raw.copy(), required_cols)

    # ordena por timestamp
    df["timestamp"] = to_numeric_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if df.empty:
        return {}, {"error": "empty_df"}, pd.DataFrame()

    t0 = float(df["timestamp"].iloc[0])
    df["t_rel_s"] = (df["timestamp"] - t0).astype(int)

    # identifica columnas gauge/counter necesarias
    gauge_cols = []
    counter_cols = []
    for mname, m in metrics_cfg.items():
        col = m["column"]
        kind = m["kind"]
        if kind.startswith("gauge"):
            if col not in gauge_cols:
                gauge_cols.append(col)
        elif kind.startswith("counter"):
            if col not in counter_cols:
                counter_cols.append(col)

    df_res = resample_df(
        df,
        resample_s,
        gauge_cols=gauge_cols,
        counter_cols=counter_cols,
        method=resample_method,
        interp_cfg=resample_interp_cfg,
    )
    if crop_t_rel_max is not None:
        df_res = df_res[df_res["t_rel_s"] <= int(crop_t_rel_max)].copy()
    if df_res.empty or "cpu_perc" not in df_res.columns:
        # si no hay cpu_perc, igual dejamos resampleado (si existe) pero warmup no auto
        pass

    # warmup
    mode = warmup_cfg.get("mode", "auto")
    warmup_s = 0
    if mode == "none":
        warmup_s = 0
    elif mode == "fixed":
        warmup_s = int(warmup_cfg.get("fixed", {}).get("seconds", 0))
    elif mode == "auto":
        auto = warmup_cfg.get("auto", {})
        w = detect_warmup_auto(
            cpu_series=df_res.get("cpu_perc", pd.Series(dtype=float)),
            t_series=df_res.get("t_rel_s", pd.Series(dtype=int)),
            cfg=auto,
        )
        if w is None:
            warmup_s = int(warmup_cfg.get("fallback_if_not_found", 0))
        else:
            warmup_s = int(w)
    else:
        warmup_s = 0

    # segmento post-warmup para resúmenes
    seg = df_res[df_res["t_rel_s"] >= warmup_s].copy()

    meta = {
        "t0_epoch": t0,
        "warmup_s": warmup_s,
        "samples_resampled": int(df_res.shape[0]),
        "samples_used": int(seg.shape[0]),
        "duration_total_s": int(df_res["t_rel_s"].max()) if not df_res.empty else None,
        "duration_used_s": int(seg["t_rel_s"].max() - seg["t_rel_s"].min()) if len(seg) >= 2 else 0,
    }

    metrics_out: Dict[str, Optional[float]] = {}

    for mname, m in metrics_cfg.items():
        kind = m["kind"]
        col = m["column"]
        transform = m.get("transform", None)
        tf = TRANSFORMS.get(transform) if transform else None

        if seg.empty:
            metrics_out[mname] = None
            continue

        if kind == "gauge_mean":
            v = to_numeric_series(seg[col])
            val = float(v.mean(skipna=True)) if v.notna().any() else None
            metrics_out[mname] = tf(val) if (val is not None and tf) else val

        elif kind == "gauge_max":
            v = to_numeric_series(seg[col])
            val = float(v.max(skipna=True)) if v.notna().any() else None
            metrics_out[mname] = tf(val) if (val is not None and tf) else val

        elif kind == "gauge_quantile":
            q = float(m.get("quantile", 0.95))
            q = min(1.0, max(0.0, q))
            v = to_numeric_series(seg[col]).dropna()
            val = float(v.quantile(q)) if not v.empty else None
            metrics_out[mname] = tf(val) if (val is not None and tf) else val

        elif kind == "counter_total":
            # delta último - primero dentro del segmento (contadores ya ffill)
            d = delta_last_first(seg[col])
            metrics_out[mname] = tf(d) if (d is not None and tf) else d

        elif kind == "counter_rate":
            # tasa por segundo en la serie resampleada:
            # diff / dt (dt = resample_s)
            c = to_numeric_series(df_res[col]).ffill()
            if c.dropna().empty:
                metrics_out[mname] = None
            else:
                rate = c.diff() / float(resample_s)
                # nos interesa promedio POST warmup para resumen
                rate_seg = rate[df_res["t_rel_s"] >= warmup_s]
                rate_val = float(rate_seg.mean(skipna=True)) if rate_seg.notna().any() else None
                metrics_out[mname] = tf(rate_val) if (rate_val is not None and tf) else rate_val

        else:
            metrics_out[mname] = None

    return metrics_out, meta, df_res


def build_timeseries_metrics(
    df_res: pd.DataFrame,
    resample_s: int,
    metrics_cfg: dict,
) -> pd.DataFrame:
    """
    Construye dataframe con columnas por métrica (por tiempo) para agregación entre reps.
    Devuelve: t_rel_s + columnas métricas (ya con transforms aplicadas).
    """
    if df_res.empty:
        return pd.DataFrame()

    out = pd.DataFrame({"t_rel_s": df_res["t_rel_s"].astype(int)})

    for mname, m in metrics_cfg.items():
        kind = m["kind"]
        col = m["column"]
        transform = m.get("transform", None)
        tf = TRANSFORMS.get(transform) if transform else None

        if kind in ("gauge_mean", "gauge_max", "gauge_quantile"):
            v = to_numeric_series(df_res.get(col, pd.Series(dtype=float)))
            out[mname] = tf(v) if tf else v

        elif kind == "counter_rate":
            c = to_numeric_series(df_res.get(col, pd.Series(dtype=float))).ffill()
            rate = c.diff() / float(resample_s)
            rate = rate.fillna(0.0)
            out[mname] = tf(rate) if tf else rate


        # Nota: counter_total no se suele graficar como serie; si lo pones igual,
        # sería acumulado. Por ahora lo omitimos en series.
        elif kind == "counter_total":
            continue

    return out


# -------------------------
# Plotting
# -------------------------

def plot_grouped_bars(
    load_df: pd.DataFrame,
    out_path: Path,
    metric: str,
    metric_label: str,
    loads_order: List[str],
    controllers_order: List[str],
    labels_cfg: Optional[dict] = None,
):
    """
    Bar plot agrupado:
      X: load
      barras dentro: controller
    Usa mean y CI si existen.
    """
    # columnas esperadas
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"

    # prepara matriz
    width = 0.8
    n_ctrl = max(1, len(controllers_order))
    bar_w = width / n_ctrl

    x = list(range(len(loads_order)))
    fig, ax = plt.subplots()
    apply_figsize(fig, labels_cfg)

    for j, ctrl in enumerate(controllers_order):
        ys = []
        yerr = [[], []]  # lower, upper
        for ld in loads_order:
            row = load_df[(load_df["controller"] == ctrl) & (load_df["load"] == ld)]
            if row.empty:
                ys.append(float("nan"))
                yerr[0].append(0.0)
                yerr[1].append(0.0)
                continue
            y = row.iloc[0].get(mean_col, float("nan"))
            lo = row.iloc[0].get(low_col, float("nan"))
            hi = row.iloc[0].get(high_col, float("nan"))
            ys.append(y)
            if pd.notna(lo) and pd.notna(hi) and pd.notna(y):
                yerr[0].append(max(0.0, y - lo))
                yerr[1].append(max(0.0, hi - y))
            else:
                yerr[0].append(0.0)
                yerr[1].append(0.0)

        offsets = [xi - width/2 + (j + 0.5)*bar_w for xi in x]
        ax.bar(offsets, ys, bar_w, yerr=yerr, capsize=3, label=ctrl)

    labels_cfg = labels_cfg or {}
    title = labels_cfg.get("title") if "title" in labels_cfg else f"{metric_label} (media ± IC95)"
    xlabel = labels_cfg.get("xlabel") if "xlabel" in labels_cfg else "Carga"
    ylabel = labels_cfg.get("ylabel") if "ylabel" in labels_cfg else metric_label
    legend_title = labels_cfg.get("legend")

    load_suffix = labels_cfg.get("load_suffix") if labels_cfg else None
    disp_loads = [format_load_label(ld, load_suffix) for ld in loads_order]
    ax.set_xticks(x)
    ax.set_xticklabels(disp_loads, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(title=legend_title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_timeseries(ts_df: pd.DataFrame, out_path: Path, metric: str, metric_label: str,
                   labels_cfg: Optional[dict] = None, show_ci: bool = True, smooth_cfg: Optional[dict] = None,
                   y_limits: Optional[Tuple[float, float]] = None, y_nbins: Optional[int] = None):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"

    df = ts_df.copy()
    df[mean_col] = smooth_series(df[mean_col], smooth_cfg)
    if show_ci and low_col in df.columns and high_col in df.columns:
        df[low_col] = smooth_series(df[low_col], smooth_cfg)
        df[high_col] = smooth_series(df[high_col], smooth_cfg)

    fig, ax = plt.subplots()
    apply_figsize(fig, labels_cfg)
    ax.plot(df["t_rel_s"], df[mean_col])
    if show_ci and low_col in df.columns and high_col in df.columns:
        ax.fill_between(df["t_rel_s"], df[low_col], df[high_col], alpha=0.2)

    labels_cfg = labels_cfg or {}
    title = labels_cfg.get("title", "")
    xlabel = labels_cfg.get("xlabel", "")
    # usa solo el ylabel provisto (sin agregar metric_label)
    ylabel = labels_cfg.get("ylabel", "")
    legend_title = labels_cfg.get("legend")
    y_nbins_val = labels_cfg.get("y_nbins", None) if labels_cfg else None
    if y_nbins is None:
        y_nbins = y_nbins_val

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    if y_nbins:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=int(y_nbins)))
    if legend_title:
        ax.legend(title=legend_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_multi(ts_df: pd.DataFrame, out_path: Path, metric: str, metric_label: str,
                          controllers_order=None, labels_cfg: Optional[dict] = None,
                          show_ci: bool = True, smooth_cfg: Optional[dict] = None,
                          y_limits: Optional[Tuple[float, float]] = None, y_nbins: Optional[int] = None):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"

    df = ts_df.sort_values(["controller", "t_rel_s"]).copy()
    df = df.dropna(subset=[mean_col])
    if df.empty:
        print(f"WARNING: no hay datos para plot {metric} -> {out_path}")
        return

    fig, ax = plt.subplots()

    controllers = controllers_order or sorted(df["controller"].dropna().unique().tolist())
    for ctrl in controllers:
        d = df[df["controller"] == ctrl].sort_values("t_rel_s")
        d[mean_col] = smooth_series(d[mean_col], smooth_cfg)
        if show_ci and low_col in d.columns and high_col in d.columns:
            d[low_col] = smooth_series(d[low_col], smooth_cfg)
            d[high_col] = smooth_series(d[high_col], smooth_cfg)
        if d.empty:
            continue

        ax.plot(d["t_rel_s"], d[mean_col], label=ctrl)

        # banda CI si existe
        if show_ci and low_col in d.columns and high_col in d.columns and d[low_col].notna().any() and d[high_col].notna().any():
            ax.fill_between(d["t_rel_s"], d[low_col], d[high_col], alpha=0.15)

    labels_cfg = labels_cfg or {}
    title = labels_cfg.get("title", "")
    xlabel = labels_cfg.get("xlabel", "")
    ylabel = labels_cfg.get("ylabel", "")
    legend_title = labels_cfg.get("legend")
    y_nbins_val = labels_cfg.get("y_nbins", None) if labels_cfg else None
    if y_nbins is None:
        y_nbins = y_nbins_val

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    if y_nbins:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=int(y_nbins)))
    if legend_title:
        ax.legend(title=legend_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_faceted(
    ts_df: pd.DataFrame,
    out_path: Path,
    metric: str,
    metric_label: str,
    controllers_order=None,
    labels_cfg: Optional[dict] = None,
    show_ci: bool = True,
    smooth_cfg: Optional[dict] = None,
    show_peaks: bool = False,
    y_limits: Optional[Tuple[float, float]] = None,
    y_nbins: Optional[int] = None,
):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"

    df = ts_df.sort_values(["controller", "t_rel_s"]).copy()
    df = df.dropna(subset=[mean_col])
    if df.empty:
        print(f"WARNING: no hay datos para plot faceted {metric} -> {out_path}")
        return

    controllers = controllers_order or sorted(df["controller"].dropna().unique().tolist())
    n_ctrl = len(controllers)
    fig, axes = plt.subplots(n_ctrl, 1, figsize=(6, max(3, 2*n_ctrl)), sharex=True)
    if n_ctrl == 1:
        axes = [axes]

    labels_cfg = labels_cfg or {}
    title = labels_cfg.get("title", "")
    xlabel = labels_cfg.get("xlabel", "")
    ylabel = labels_cfg.get("ylabel", "")
    grid = bool(labels_cfg.get("grid", False))
    y_nbins_val = labels_cfg.get("y_nbins", None) if labels_cfg else None
    if y_nbins is None:
        y_nbins = y_nbins_val

    for ax, ctrl in zip(axes, controllers):
        d = df[df["controller"] == ctrl].sort_values("t_rel_s")
        if d.empty:
            ax.set_title(ctrl)
            ax.set_ylabel(ylabel)
            if grid:
                ax.grid(True, linestyle="--", alpha=0.4)
            if y_limits is not None:
                ax.set_ylim(y_limits)
            if y_nbins:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=int(y_nbins)))
            continue
        d[mean_col] = smooth_series(d[mean_col], smooth_cfg)
        if show_ci and low_col in d.columns and high_col in d.columns:
            d[low_col] = smooth_series(d[low_col], smooth_cfg)
            d[high_col] = smooth_series(d[high_col], smooth_cfg)

        ax.plot(d["t_rel_s"], d[mean_col], label=metric_label)
        if show_ci and low_col in d.columns and high_col in d.columns and d[low_col].notna().any() and d[high_col].notna().any():
            ax.fill_between(d["t_rel_s"], d[low_col], d[high_col], alpha=0.15)

        if show_peaks and d[mean_col].notna().any():
            idx_peak = d[mean_col].idxmax()
            if pd.notna(idx_peak):
                t_peak = d.loc[idx_peak, "t_rel_s"]
                y_peak = d.loc[idx_peak, mean_col]
                ax.scatter([t_peak], [y_peak], color="red", marker="o", zorder=5)

        ax.set_title(f"{ctrl}")
        ax.set_ylabel(ylabel or "")
        if grid:
            ax.grid(True, linestyle="--", alpha=0.4)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        if y_nbins:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=int(y_nbins)))

    axes[-1].set_xlabel(xlabel)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



# -------------------------
# Main builder
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reports_containers.yml")
    ap.add_argument("--regenerate-csvs", action="store_true", help="Recalcula y reescribe CSVs aunque ya existan.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    extracted_root = Path(cfg["inputs"]["extracted_root"]).resolve()
    glob_pat = cfg["inputs"]["containers_glob"]
    container_filter = cfg["inputs"].get("container_filter", []) or []
    required_cols = cfg["inputs"].get("required_columns", []) or []

    resample_s = int(cfg["time"]["resample_s"])
    resample_method = cfg["time"].get("resample_method", "sample_hold")
    resample_interp_cfg = cfg["time"].get("interpolation", {})
    min_reps_per_t = int(cfg["time"].get("min_reps_per_t", 1))
    crop_common = bool(cfg["time"].get("crop_to_common_time", False))

    warmup_cfg = cfg["warmup"]
    metrics_cfg = cfg["metrics"]

    out_dir = Path(cfg["outputs"]["out_dir"]).resolve()
    plots_dir = Path(cfg["outputs"]["plots_dir"]).resolve()
    safe_mkdir(out_dir)
    safe_mkdir(plots_dir)

    rep_summary_path = out_dir / cfg["outputs"]["csv"]["rep_summary"]
    load_summary_path = out_dir / cfg["outputs"]["csv"]["load_summary"]
    timeseries_path = out_dir / cfg["outputs"]["csv"]["timeseries_mean"]
    missing_path = out_dir / cfg["outputs"]["csv"]["missing"]

    required_csvs = [rep_summary_path, load_summary_path, timeseries_path]
    use_existing_csvs = (not args.regenerate_csvs and all(p.exists() for p in required_csvs))
    if use_existing_csvs:
        print("INFO: usando CSVs existentes (usa --regenerate-csvs para regenerarlos).")
        rep_df = read_csv_maybe_empty(rep_summary_path)
        load_df = read_csv_maybe_empty(load_summary_path)
        ts_mean_df = read_csv_maybe_empty(timeseries_path)
        miss_df = read_csv_maybe_empty(missing_path)
    else:
        missing_rows = []
        rep_rows = []
        ts_rows = []
        rep_ts_meta: List[dict] = []
    
        if not extracted_root.exists():
            print(f"ERROR: extracted_root no existe: {extracted_root}")
            return
    
        controllers = sorted([p for p in extracted_root.iterdir() if p.is_dir()])
        if not controllers:
            print("WARNING: no se encontraron controladores en extracted_root.")
            return
    
        for ctrl_dir in controllers:
            controller = ctrl_dir.name
            loads = sorted([p for p in ctrl_dir.iterdir() if p.is_dir()], key=lambda p: natural_load_key(p.name))
    
            for load_dir in loads:
                load = load_dir.name
                # busca todos los csv de contenedores (por rep)
                csvs = list(load_dir.glob(glob_pat))
                if not csvs:
                    missing_rows.append({
                        "controller": controller, "load": load, "rep": None,
                        "container": None, "path": str(load_dir), "reason": "no_csvs_found"
                    })
                    continue
    
                for csv_path in sorted(csvs):
                    # parse rep_?
                    mrep = re.search(r"(rep_\d+)", str(csv_path))
                    rep = mrep.group(1) if mrep else "rep_unknown"
                    container = csv_path.stem
    
                    if container_filter and container not in container_filter:
                        continue
    
                    try:
                        df_raw = pd.read_csv(csv_path)
                    except Exception as e:
                        missing_rows.append({
                            "controller": controller, "load": load, "rep": rep,
                            "container": container, "path": str(csv_path), "reason": f"read_error:{e}"
                        })
                        continue
    
                    if df_raw.empty:
                        missing_rows.append({
                            "controller": controller, "load": load, "rep": rep,
                            "container": container, "path": str(csv_path), "reason": "empty_csv"
                        })
                        continue
    
                    # resumen por rep (post-warmup)
                    # cropping se aplica luego si es requerido
                    metrics_out, meta, df_res = compute_rep_summary(
                        df_raw=df_raw,
                        resample_s=resample_s,
                        resample_method=resample_method,
                        resample_interp_cfg=resample_interp_cfg,
                        warmup_cfg=warmup_cfg,
                        metrics_cfg=metrics_cfg,
                        required_cols=required_cols,
                        crop_t_rel_max=None,
                    )
                    if "error" in meta:
                        missing_rows.append({
                            "controller": controller, "load": load, "rep": rep,
                            "container": container, "path": str(csv_path), "reason": meta["error"]
                        })
                        continue
    
                    row = {
                        "controller": controller,
                        "load": load,
                        "rep": rep,
                        "container": container,
                        "src_csv": str(csv_path),
                        **meta,
                        **metrics_out,
                    }
                    rep_rows.append(row)

                    # series de tiempo por rep (incluye warmup)
                    df_ts = build_timeseries_metrics(df_res, resample_s=resample_s, metrics_cfg=metrics_cfg)
                    if not df_ts.empty:
                        for _, r in df_ts.iterrows():
                            ts_rows.append({
                                "controller": controller,
                                "load": load,
                                "rep": rep,
                                "container": container,
                                "t_rel_s": int(r["t_rel_s"]),
                                **{k: r.get(k, None) for k in df_ts.columns if k != "t_rel_s"},
                            })
                    rep_ts_meta.append({
                        "controller": controller,
                        "load": load,
                        "rep": rep,
                        "container": container,
                        "t_max": int(df_res["t_rel_s"].max()) if not df_res.empty else 0,
                        "csv_path": str(csv_path),
                    })
    
        rep_df = pd.DataFrame(rep_rows)
        miss_df = pd.DataFrame(missing_rows)
        ts_rep_df = pd.DataFrame(ts_rows)
        meta_ts_df = pd.DataFrame(rep_ts_meta)

        if crop_common and not ts_rep_df.empty and not meta_ts_df.empty:
            # calcula duración mínima por controller/load/container
            mins = (
                meta_ts_df.groupby(["controller", "load", "container"])["t_max"]
                .min()
                .reset_index()
                .rename(columns={"t_max": "t_max_min"})
            )
            ts_rep_df = ts_rep_df.merge(mins, on=["controller", "load", "container"], how="left")
            ts_rep_df = ts_rep_df[ts_rep_df["t_rel_s"] <= ts_rep_df["t_max_min"]].drop(columns=["t_max_min"])

        rep_df.to_csv(rep_summary_path, index=False)
        miss_df.to_csv(missing_path, index=False)
    
        # ----------------- load_summary (agrega sobre reps) -----------------
        conf = cfg.get("stats", {}).get("confidence", {})
        ci_enable = bool(conf.get("enable", True))
        ci_level = float(conf.get("level", 0.95))
        ci_method = conf.get("method", "t")
    
        metric_names = list(metrics_cfg.keys())
    
        load_rows = []
        if not rep_df.empty:
            grp = rep_df.groupby(["controller", "load", "container"], dropna=False)
            for (controller, load, container), g in grp:
                out = {"controller": controller, "load": load, "container": container}
                for m in metric_names:
                    v = pd.to_numeric(g[m], errors="coerce").dropna()
                    n = int(v.shape[0])
                    mean = float(v.mean()) if n else None
                    std = float(v.std(ddof=1)) if n >= 2 else None
    
                    out[f"{m}_mean"] = mean
                    out[f"{m}_std"] = std
                    out[f"{m}_min"] = float(v.min()) if n else None
                    out[f"{m}_max"] = float(v.max()) if n else None
                    out[f"{m}_count"] = n

                    if ci_enable and (n >= 2) and (mean is not None) and (std is not None):
                        lo, hi = compute_ci(mean, std, n, ci_level, ci_method)
                        lo, hi = clamp_ci(lo, hi)
                        out[f"{m}_ci95_low"] = lo
                        out[f"{m}_ci95_high"] = hi
                    else:
                        out[f"{m}_ci95_low"] = None
                        out[f"{m}_ci95_high"] = None
                load_rows.append(out)
    
        load_df = pd.DataFrame(load_rows)
        if not load_df.empty:
            # orden por carga
            load_df["__load_key"] = load_df["load"].apply(natural_load_key)
            load_df = load_df.sort_values(["controller", "__load_key", "container"]).drop(columns=["__load_key"])
        load_df.to_csv(load_summary_path, index=False)
    
        # ----------------- timeseries_mean (agrega sobre reps por t) -----------------
        ts_rows_out = []
        if not ts_rep_df.empty:
            # agrega por controller/load/container/t_rel_s
            g2 = ts_rep_df.groupby(["controller", "load", "container", "t_rel_s"], dropna=False)
            for keys, g in g2:
                controller, load, container, t_rel_s = keys
                out = {"controller": controller, "load": load, "container": container, "t_rel_s": int(t_rel_s)}
                for m in metric_names:
                    if m not in g.columns:
                        continue
                    v = pd.to_numeric(g[m], errors="coerce").dropna()
                    n = int(v.shape[0])
                    mean = float(v.mean()) if n else None
                    std = float(v.std(ddof=1)) if n >= 2 else None
    
                    out[f"{m}_mean"] = mean
                    out[f"{m}_std"] = std
                    out[f"{m}_count"] = n

                    if ci_enable and (n >= 2) and (mean is not None) and (std is not None):
                        lo, hi = compute_ci(mean, std, n, ci_level, ci_method)
                        lo, hi = clamp_ci(lo, hi)
                        out[f"{m}_ci95_low"] = lo
                        out[f"{m}_ci95_high"] = hi
                    else:
                        out[f"{m}_ci95_low"] = None
                        out[f"{m}_ci95_high"] = None
    
                ts_rows_out.append(out)
    
        ts_mean_df = pd.DataFrame(ts_rows_out)
        if not ts_mean_df.empty:
            ts_mean_df["__load_key"] = ts_mean_df["load"].apply(natural_load_key)
            ts_mean_df = ts_mean_df.sort_values(["controller", "__load_key", "container", "t_rel_s"]).drop(columns=["__load_key"])
        ts_mean_df.to_csv(timeseries_path, index=False)
    
        print(f"OK: {rep_summary_path}")
        print(f"OK: {load_summary_path}")
        print(f"OK: {timeseries_path}")
        print(f"OK: {missing_path}")
    
    # ----------------- Plots -----------------
    plots_cfg = cfg.get("plots", {})
    metric_label_map = {m: metric_display_name(metrics_cfg, m) for m in metrics_cfg.keys()}
    def compute_y_limits(df: pd.DataFrame, metric: str, show_ci: bool) -> Optional[Tuple[float, float]]:
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci95_low"
        high_col = f"{metric}_ci95_high"
        if df.empty or mean_col not in df.columns:
            return None
        vals = pd.to_numeric(df[mean_col], errors="coerce")
        if show_ci and high_col in df.columns:
            vals = pd.concat([vals, pd.to_numeric(df[high_col], errors="coerce")], ignore_index=True)
        if show_ci and low_col in df.columns:
            vals = pd.concat([vals, pd.to_numeric(df[low_col], errors="coerce")], ignore_index=True)
        vals = vals.dropna()
        if vals.empty:
            return None
        return (float(vals.min()), float(vals.max()))

    if plots_cfg.get("bar_by_load", {}).get("enable", False) and not load_df.empty:
        bar_cfg = plots_cfg["bar_by_load"]
        container_sel = bar_cfg.get("container", None)
        metrics_sel = bar_cfg.get("metrics", [])
        labels_cfg = bar_cfg.get("labels", {})
        if "load_suffix" not in labels_cfg and "load_suffix" in bar_cfg:
            labels_cfg["load_suffix"] = bar_cfg.get("load_suffix")
        loads_sel = bar_cfg.get("loads", None)
        dfp = load_df.copy()
        if container_sel:
            dfp = dfp[dfp["container"] == container_sel]

        if isinstance(loads_sel, str) and loads_sel == "all":
            loads_sel = None
        if loads_sel is not None:
            # filtra y respeta el orden provisto
            dfp = dfp[dfp["load"].isin(loads_sel)]
            loads_order = list(loads_sel)
        else:
            loads_order = sorted(dfp["load"].dropna().unique().tolist(), key=natural_load_key)
        controllers_order = sorted(dfp["controller"].dropna().unique().tolist())

        for m in metrics_sel:
            outp = plots_dir / f"bar_{m}.png"
            plot_grouped_bars(
                dfp,
                outp,
                metric=m,
                metric_label=metric_label_map.get(m, m),
                loads_order=loads_order,
                controllers_order=controllers_order,
                labels_cfg=labels_cfg,
            )
            print(f"OK plot: {outp}")

    if plots_cfg.get("timeseries_avg_reps", {}).get("enable", False) and not ts_mean_df.empty:
        ts_cfg = plots_cfg["timeseries_avg_reps"]
        sel = ts_cfg.get("select", {})
        metrics_sel = ts_cfg.get("metrics", [])
        labels_cfg = ts_cfg.get("labels", {})
        show_ci = bool(ts_cfg.get("show_ci", True))
        smooth_cfg = ts_cfg.get("smooth", {})
        y_limits = None
        y_nbins = labels_cfg.get("y_nbins") if isinstance(labels_cfg, dict) else None

        dfp = ts_mean_df.copy()
        for k in ["controller", "load", "container"]:
            if sel.get(k) is not None:
                dfp = dfp[dfp[k] == sel[k]]

        # filtra tiempos con suficientes reps (solo para graficar)
        # usa count de la primera métrica si existe
        if metrics_sel:
            ccol = f"{metrics_sel[0]}_count"
            if ccol in dfp.columns:
                dfp = dfp[dfp[ccol] >= min_reps_per_t]

        for m in metrics_sel:
            if y_limits is None:
                y_limits = compute_y_limits(dfp, m, show_ci)
            outp = plots_dir / f"ts_{sel.get('controller','')}_{sel.get('load','')}_{m}.png"
            plot_timeseries(
                dfp,
                outp,
                metric=m,
                metric_label=metric_label_map.get(m, m),
                labels_cfg=labels_cfg,
                show_ci=show_ci,
                smooth_cfg=smooth_cfg,
                y_limits=y_limits,
                y_nbins=y_nbins,
            )
            print(f"OK plot: {outp}")

    if plots_cfg.get("timeseries_multi_controllers", {}).get("enable", False) and not ts_mean_df.empty:
        ts_cfg = plots_cfg["timeseries_multi_controllers"]
        sel = ts_cfg.get("select", {})
        metrics_sel = ts_cfg.get("metrics", [])
        min_reps_plot = int(ts_cfg.get("min_reps_per_t", min_reps_per_t))
        labels_cfg = ts_cfg.get("labels", {})
        show_ci = bool(ts_cfg.get("show_ci", True))
        smooth_cfg = ts_cfg.get("smooth", {})
        y_nbins = labels_cfg.get("y_nbins") if isinstance(labels_cfg, dict) else None

        dfp = ts_mean_df.copy()

        # Filtros: load y container
        if sel.get("load") is not None:
            dfp = dfp[dfp["load"] == sel["load"]]
        if sel.get("container") is not None:
            dfp = dfp[dfp["container"] == sel["container"]]

        # Controladores: all o lista
        ctrls = sel.get("controllers", "all")
        if isinstance(ctrls, list):
            dfp = dfp[dfp["controller"].isin(ctrls)]

        # Filtra puntos con suficientes reps (por controlador y tiempo)
        if metrics_sel:
            ccol = f"{metrics_sel[0]}_count"
            if ccol in dfp.columns:
                dfp = dfp[dfp[ccol] >= min_reps_plot]

        # Plotea una figura por métrica (multi-curva)
        load_label = sel.get("load", "load")
        cont_label = sel.get("container", "container")
        controllers_plot = sorted(dfp["controller"].dropna().unique().tolist())

        for m in metrics_sel:
            y_limits = compute_y_limits(dfp, m, show_ci)
            outp = plots_dir / f"ts_multi_{load_label}_{cont_label}_{m}.png"
            plot_timeseries_multi(
                dfp,
                outp,
                metric=m,
                metric_label=metric_label_map.get(m, m),
                controllers_order=controllers_plot,
                labels_cfg=labels_cfg,
                show_ci=show_ci,
                smooth_cfg=smooth_cfg,
                y_limits=y_limits,
                y_nbins=y_nbins,
            )
            print(f"OK plot: {outp}")

    if plots_cfg.get("timeseries_multi_controllers_faceted", {}).get("enable", False) and not ts_mean_df.empty:
        ts_cfg = plots_cfg["timeseries_multi_controllers_faceted"]
        sel = ts_cfg.get("select", {})
        metrics_sel = ts_cfg.get("metrics", [])
        min_reps_plot = int(ts_cfg.get("min_reps_per_t", min_reps_per_t))
        labels_cfg = ts_cfg.get("labels", {})
        show_ci = bool(ts_cfg.get("show_ci", True))
        smooth_cfg = ts_cfg.get("smooth", {})
        show_peaks = bool(ts_cfg.get("show_peaks", False))
        y_nbins = labels_cfg.get("y_nbins") if isinstance(labels_cfg, dict) else None

        dfp = ts_mean_df.copy()

        # Filtros: load y container
        if sel.get("load") is not None:
            dfp = dfp[dfp["load"] == sel["load"]]
        if sel.get("container") is not None:
            dfp = dfp[dfp["container"] == sel["container"]]

        # Controladores: all o lista
        ctrls = sel.get("controllers", "all")
        if isinstance(ctrls, list):
            dfp = dfp[dfp["controller"].isin(ctrls)]

        # Filtra puntos con suficientes reps (por controlador y tiempo)
        if metrics_sel:
            ccol = f"{metrics_sel[0]}_count"
            if ccol in dfp.columns:
                dfp = dfp[dfp[ccol] >= min_reps_plot]

        controllers_plot = sorted(dfp["controller"].dropna().unique().tolist())
        load_label = sel.get("load", "load")
        cont_label = sel.get("container", "container")

        for m in metrics_sel:
            y_limits = compute_y_limits(dfp, m, show_ci)
            outp = plots_dir / f"ts_multi_faceted_{load_label}_{cont_label}_{m}.png"
            plot_timeseries_faceted(
                dfp,
                outp,
                metric=m,
                metric_label=metric_label_map.get(m, m),
                controllers_order=controllers_plot,
                labels_cfg=labels_cfg,
                show_ci=show_ci,
                smooth_cfg=smooth_cfg,
                show_peaks=show_peaks,
                y_limits=y_limits,
                y_nbins=y_nbins,
            )
            print(f"OK plot: {outp}")



if __name__ == "__main__":
    main()
