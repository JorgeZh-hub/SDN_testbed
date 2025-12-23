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


# -------------------------
# Helpers
# -------------------------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def natural_load_key(s: str):
    # "1a" -> 1.0, "1.2a" -> 1.2, fallback: string
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return s

def bytes_to_mib(x: float) -> float:
    return x / (1024.0 * 1024.0)

def bytes_to_kib(x: float) -> float:
    return x / 1024.0

TRANSFORMS = {
    "bytes_to_mib": bytes_to_mib,
    "bytes_to_kib": bytes_to_kib,
}

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

def resample_df(df: pd.DataFrame, resample_s: int, gauge_cols: List[str], counter_cols: List[str]) -> pd.DataFrame:
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
    warmup_cfg: dict,
    metrics_cfg: dict,
    required_cols: List[str],
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

    df_res = resample_df(df, resample_s, gauge_cols=gauge_cols, counter_cols=counter_cols)
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

        if kind == "gauge_mean":
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

def plot_grouped_bars(load_df: pd.DataFrame, out_path: Path, metric: str, loads_order: List[str], controllers_order: List[str]):
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

    ax.set_xticks(x)
    ax.set_xticklabels(loads_order, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (mean ± CI95)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_timeseries(ts_df: pd.DataFrame, out_path: Path, metric: str):
    mean_col = f"{metric}_mean"
    low_col  = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"

    fig, ax = plt.subplots()
    ax.plot(ts_df["t_rel_s"], ts_df[mean_col], label="mean")
    if low_col in ts_df.columns and high_col in ts_df.columns:
        ax.fill_between(ts_df["t_rel_s"], ts_df[low_col], ts_df[high_col], alpha=0.2, label="CI95")

    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel(metric)
    ax.set_title(f"Time series: {metric} (mean over reps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_timeseries_multi(ts_df: pd.DataFrame, out_path: Path, metric: str, controllers_order=None):
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
        if d.empty:
            continue

        ax.plot(d["t_rel_s"], d[mean_col], label=ctrl)

        # banda CI si existe
        if low_col in d.columns and high_col in d.columns and d[low_col].notna().any() and d[high_col].notna().any():
            ax.fill_between(d["t_rel_s"], d[low_col], d[high_col], alpha=0.15)

    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: curvas por controlador")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



# -------------------------
# Main builder
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reports_containers.yml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    extracted_root = Path(cfg["inputs"]["extracted_root"]).resolve()
    glob_pat = cfg["inputs"]["containers_glob"]
    container_filter = cfg["inputs"].get("container_filter", []) or []
    required_cols = cfg["inputs"].get("required_columns", []) or []

    resample_s = int(cfg["time"]["resample_s"])
    min_reps_per_t = int(cfg["time"].get("min_reps_per_t", 1))

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

    missing_rows = []
    rep_rows = []
    ts_rows = []

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
                metrics_out, meta, df_res = compute_rep_summary(
                    df_raw=df_raw,
                    resample_s=resample_s,
                    warmup_cfg=warmup_cfg,
                    metrics_cfg=metrics_cfg,
                    required_cols=required_cols,
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

    rep_df = pd.DataFrame(rep_rows)
    miss_df = pd.DataFrame(missing_rows)
    ts_rep_df = pd.DataFrame(ts_rows)

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
    if plots_cfg.get("bar_by_load", {}).get("enable", False) and not load_df.empty:
        bar_cfg = plots_cfg["bar_by_load"]
        container_sel = bar_cfg.get("container", None)
        metrics_sel = bar_cfg.get("metrics", [])
        dfp = load_df.copy()
        if container_sel:
            dfp = dfp[dfp["container"] == container_sel]

        loads_order = sorted(dfp["load"].dropna().unique().tolist(), key=natural_load_key)
        controllers_order = sorted(dfp["controller"].dropna().unique().tolist())

        for m in metrics_sel:
            outp = plots_dir / f"bar_{m}.png"
            plot_grouped_bars(dfp, outp, metric=m, loads_order=loads_order, controllers_order=controllers_order)
            print(f"OK plot: {outp}")

    if plots_cfg.get("timeseries_avg_reps", {}).get("enable", False) and not ts_mean_df.empty:
        ts_cfg = plots_cfg["timeseries_avg_reps"]
        sel = ts_cfg.get("select", {})
        metrics_sel = ts_cfg.get("metrics", [])

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
            outp = plots_dir / f"ts_{sel.get('controller','')}_{sel.get('load','')}_{m}.png"
            plot_timeseries(dfp, outp, metric=m)
            print(f"OK plot: {outp}")

    if plots_cfg.get("timeseries_multi_controllers", {}).get("enable", False) and not ts_mean_df.empty:
        ts_cfg = plots_cfg["timeseries_multi_controllers"]
        sel = ts_cfg.get("select", {})
        metrics_sel = ts_cfg.get("metrics", [])
        min_reps_plot = int(ts_cfg.get("min_reps_per_t", min_reps_per_t))

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

        for m in metrics_sel:
            outp = plots_dir / f"ts_multi_{load_label}_{cont_label}_{m}.png"
            plot_timeseries_multi(dfp, outp, metric=m)
            print(f"OK plot: {outp}")



if __name__ == "__main__":
    main()
