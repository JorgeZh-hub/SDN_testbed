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


# -----------------------------
# Utilities
# -----------------------------

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def parse_ip_port(token: str) -> Tuple[Optional[str], Optional[int]]:
    """
    token: "198.51.100.82:58806" o "198.51.100.82:-"
    """
    if token is None:
        return None, None
    if ":" not in token:
        # podría ser solo IP
        return token, None
    ip, port = token.split(":", 1)
    if port == "-" or port == "":
        return ip, None
    try:
        return ip, int(port)
    except Exception:
        return ip, None


def parse_path_list(path_text: str) -> List[int]:
    """
    "6, 5, 2, 1, 3" -> [6,5,2,1,3]
    """
    if path_text is None:
        return []
    nums: List[int] = []
    for x in path_text.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            nums.append(int(x))
        except Exception:
            pass
    return nums


def t_confidence_interval(values: List[float], level: float = 0.95) -> Tuple[float, float]:
    """
    CI para la media usando aproximación t.
    Si n<2, devuelve (mean, mean).

    Nota: sin SciPy, usamos una tabla compacta de t críticos para 95%.
    """
    import statistics
    if not values:
        return (0.0, 0.0)
    mean = statistics.mean(values)
    if len(values) < 2:
        return (mean, mean)

    stdev = statistics.stdev(values)
    n = len(values)
    df = n - 1

    # Tabla compacta t_{0.975, df} (IC95%)
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042,
    }
    if df in t_table:
        tcrit = t_table[df]
    else:
        # Aproximación por tramos
        if df < 1:
            tcrit = 12.706
        elif df < 10:
            tcrit = 2.228
        elif df < 20:
            tcrit = 2.093
        elif df < 30:
            tcrit = 2.042
        else:
            tcrit = 1.96

    half = float(tcrit) * (stdev / math.sqrt(n))
    return (mean - half, mean + half)


def infer_controller_load_rep(log_path: Path) -> Tuple[str, str, str]:
    """
    Intenta inferir:
      extracted_root/<controller>/<load>/rep_*/logs/controller*.log
    """
    rep = "unknown_rep"
    load = "unknown_load"
    controller = "unknown_controller"

    rep_dir = None
    for p in log_path.parents:
        if p.name.startswith("rep_"):
            rep_dir = p
            rep = p.name
            break

    if rep_dir is not None:
        load_dir = rep_dir.parent
        load = load_dir.name
        controller_dir = load_dir.parent
        controller = controller_dir.name

    return controller, load, rep


def natural_load_key(s: str):
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", str(s))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return m.group(1)
    return str(s)

def format_load_label(load_val: str, suffix: Optional[str]) -> str:
    """
    Permite reemplazar el sufijo de la carga (p.ej., '2a' -> '2n' si suffix='n').
    Si no matchea número+letras, concatena el sufijo al final.
    """
    if not suffix:
        return str(load_val)
    s = str(load_val)
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([A-Za-z]+)?$", s)
    if m:
        return f"{m.group(1)}{suffix}"
    return f"{s}{suffix}"


# -----------------------------
# Parsing config
# -----------------------------

@dataclass
class Patterns:
    class_new_cookie: re.Pattern
    flow_install: re.Pattern
    te_reroute: re.Pattern


def build_patterns(cfg: dict) -> Patterns:
    pcfg = cfg.get("parse", {}).get("patterns", {})

    # Patrones por defecto (tolerantes)
    class_pat = pcfg.get(
        "class_new_cookie",
        r"^\[CLASS\]\s+new_cookie=(0x[0-9a-fA-F]+)\s+class=([A-Z0-9\-_]+)\s+rule=([A-Z0-9\-_]+)\s+stable=([^\s]+)\s+queue=(\d+).*?(\d+\.\d+\.\d+\.\d+:\S+)\s+->\s+(\d+\.\d+\.\d+\.\d+:\S+)\s+proto=(\d+)"
    )
    flow_pat = pcfg.get(
        "flow_install",
        r"^\[FLOW\]\s+install\s+cookie=(0x[0-9a-fA-F]+)\s+class=([A-Z0-9\-_]+)\s+stable=([^\s]+)\s+queue=(\d+)\s+src=([^\s]+)\s+dst=([^\s]+)\s+proto=(\d+)\s+path=\[([0-9,\s]+)\]"
    )
    reroute_pat = pcfg.get(
        "te_reroute",
        r"^\[TE\]\s+reroute\s+cookie=(0x[0-9a-fA-F]+)->(0x[0-9a-fA-F]+)\s+class=([A-Z0-9\-_]+)\s+rate=([0-9\.]+)\s+old_path=\[([0-9,\s]+)\]\s+new_path=\[([0-9,\s]+)\]"
    )

    return Patterns(
        class_new_cookie=re.compile(class_pat),
        flow_install=re.compile(flow_pat),
        te_reroute=re.compile(reroute_pat),
    )


def detect_warmup_fixed(cfg: dict) -> int:
    return int(cfg.get("fixed", {}).get("seconds", 0))


def detect_warmup_auto(flow_rate: pd.Series, cfg: dict) -> Optional[int]:
    """
    Auto-warmup para logs:
      - usa installs/s por segundo
      - encuentra el primer t donde installs/s suavizado cae <= umbral,
        sostenido por sustain_s.
    """
    auto = cfg.get("auto", {})
    th = float(auto.get("flow_rate_th", 3.0))
    smooth_window_s = int(auto.get("smooth_window_s", 5))
    sustain_s = int(auto.get("sustain_s", 10))
    max_search_s = int(auto.get("max_search_s", 120))

    if flow_rate.empty:
        return 0

    x = flow_rate.fillna(0.0).astype(float)

    win = max(1, smooth_window_s)
    x_smooth = x.rolling(window=win, min_periods=1, center=True).mean()

    max_idx = min(int(max_search_s), int(x_smooth.index.max()))
    for t in range(0, max_idx + 1):
        t2 = t + sustain_s
        if t2 > max_idx:
            break
        seg = x_smooth.loc[t:t2]
        if (seg <= th).all():
            return int(t)

    return None


def compute_warmup_seconds(timeseries: pd.DataFrame, cfg: dict) -> int:
    warmup = cfg.get("warmup", {})
    mode = warmup.get("mode", "fixed")
    fallback = int(warmup.get("fallback_if_not_found", 0))

    if mode == "none":
        return 0
    if mode == "fixed":
        return detect_warmup_fixed(warmup)
    if mode == "auto":
        s = timeseries.set_index("t_s")["installs"]
        w = detect_warmup_auto(s, warmup)
        return int(w) if w is not None else fallback
    return 0


# -----------------------------
# Log parsing
# -----------------------------

def parse_log_file(log_path: Path, cfg: dict, patterns: Patterns) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Retorna:
      events_df: flow_install / reroute por fila
      timeseries_df: por segundo (installs, reroutes), rellenado con ceros
      rep_info: resumen de conteos (incluye post-warmup)
    """
    controller, load, rep = infer_controller_load_rep(log_path)

    # clocks para inferir tiempo (segundos relativos)
    ltime = cfg.get("log_time", {})
    prefix_clocks_cfg: Dict[str, int] = ltime.get("prefix_clocks", {"[NET]": 10, "[TE]": 5})
    first_tick_at_zero = bool(ltime.get("first_tick_at_zero", True))

    clocks = {pfx: 0 for pfx in prefix_clocks_cfg.keys()}
    seen = {pfx: False for pfx in prefix_clocks_cfg.keys()}
    current_t = 0

    # para latencia CLASS->FLOW
    class_info: Dict[str, dict] = {}
    class_seen_install: Dict[str, bool] = {}

    # cookie -> flow info (para asociar reroute)
    flow_by_cookie: Dict[str, dict] = {}

    events: List[dict] = []
    line_no = 0

    encoding = cfg.get("inputs", {}).get("encoding", "utf-8")
    errors = cfg.get("inputs", {}).get("encoding_errors", "replace")

    with log_path.open("r", encoding=encoding, errors=errors) as f:
        for raw in f:
            line_no += 1
            line = raw.strip()
            if not line:
                continue

            # actualizar relojes
            for pfx, step in prefix_clocks_cfg.items():
                if line.startswith(pfx):
                    if not seen[pfx]:
                        seen[pfx] = True
                        if not first_tick_at_zero:
                            clocks[pfx] += int(step)
                    else:
                        clocks[pfx] += int(step)

            current_t = max(clocks.values()) if any(seen.values()) else 0

            # [CLASS] new_cookie ...
            m = patterns.class_new_cookie.match(line)
            if m:
                cookie = m.group(1)
                cls = m.group(2)
                rule = m.group(3)
                stable = m.group(4)
                queue = int(m.group(5))
                src_tok = m.group(6)
                dst_tok = m.group(7)
                proto = int(m.group(8))
                src_ip, src_port = parse_ip_port(src_tok)
                dst_ip, dst_port = parse_ip_port(dst_tok)
                class_info[cookie] = dict(
                    t_s=int(current_t),
                    class_name=cls,
                    rule=rule,
                    stable=stable,
                    queue=queue,
                    src_ip=src_ip,
                    src_port=src_port,
                    dst_ip=dst_ip,
                    dst_port=dst_port,
                    proto=proto,
                )
                class_seen_install[cookie] = False
                continue

            # [TE] reroute ...
            m = patterns.te_reroute.match(line)
            if m:
                old_cookie = m.group(1)
                new_cookie = m.group(2)
                cls = m.group(3)
                rate = float(m.group(4))
                old_path = parse_path_list(m.group(5))
                new_path = parse_path_list(m.group(6))

                flow = flow_by_cookie.get(old_cookie, {})
                events.append(
                    dict(
                        controller=controller,
                        load=load,
                        rep=rep,
                        log_file=str(log_path),
                        line_no=line_no,
                        t_s=int(current_t),
                        event="reroute",
                        class_name=cls,
                        rate=rate,
                        cookie=None,
                        old_cookie=old_cookie,
                        new_cookie=new_cookie,
                        stable=None,
                        queue=None,
                        rule=None,
                        src_ip=flow.get("src_ip"),
                        src_port=flow.get("src_port"),
                        dst_ip=flow.get("dst_ip"),
                        dst_port=flow.get("dst_port"),
                        proto=flow.get("proto"),
                        path=None,
                        old_path=old_path,
                        new_path=new_path,
                        install_latency_s=None,
                        is_initial=False,
                    )
                )

                # actualiza mapping cookie -> flow
                if flow:
                    flow2 = dict(flow)
                    flow2["last_path"] = new_path
                    flow_by_cookie[new_cookie] = flow2

                continue

            # [FLOW] install ...
            m = patterns.flow_install.match(line)
            if m:
                cookie = m.group(1)
                cls = m.group(2)
                stable = m.group(3)
                queue = int(m.group(4))
                src_tok = m.group(5)
                dst_tok = m.group(6)
                proto = int(m.group(7))
                path_list = parse_path_list(m.group(8))

                src_ip, src_port = parse_ip_port(src_tok)
                dst_ip, dst_port = parse_ip_port(dst_tok)

                is_initial = False
                install_latency_s = None
                rule = None

                if cookie in class_info and not class_seen_install.get(cookie, False):
                    is_initial = True
                    class_seen_install[cookie] = True
                    install_latency_s = int(current_t) - int(class_info[cookie]["t_s"])
                    rule = class_info[cookie].get("rule")

                events.append(
                    dict(
                        controller=controller,
                        load=load,
                        rep=rep,
                        log_file=str(log_path),
                        line_no=line_no,
                        t_s=int(current_t),
                        event="flow_install",
                        class_name=cls,
                        rate=None,
                        cookie=cookie,
                        old_cookie=None,
                        new_cookie=None,
                        stable=stable,
                        queue=queue,
                        rule=rule,
                        src_ip=src_ip,
                        src_port=src_port,
                        dst_ip=dst_ip,
                        dst_port=dst_port,
                        proto=proto,
                        path=path_list,
                        old_path=None,
                        new_path=None,
                        install_latency_s=install_latency_s,
                        is_initial=is_initial,
                    )
                )

                flow_by_cookie[cookie] = dict(
                    src_ip=src_ip,
                    src_port=src_port,
                    dst_ip=dst_ip,
                    dst_port=dst_port,
                    proto=proto,
                    last_path=path_list,
                )
                continue

    events_df = pd.DataFrame(events)

    if events_df.empty:
        timeseries_df = pd.DataFrame({"t_s": [], "installs": [], "reroutes": [], "t_rel_s": []})
        rep_info = dict(
            controller=controller,
            load=load,
            rep=rep,
            log_file=str(log_path),
            warmup_s=0,
            t_max_s=0,
            installs_total=0,
            installs_post_warmup=0,
            initial_installs_total=0,
            initial_installs_post_warmup=0,
            reroutes_total=0,
            reroutes_post_warmup=0,
        )
        return events_df, timeseries_df, rep_info

    tmax = int(events_df["t_s"].max())
    t_index = pd.Index(range(0, tmax + 1), name="t_s")

    installs = (
        events_df[events_df["event"] == "flow_install"]
        .groupby("t_s")
        .size()
        .reindex(t_index, fill_value=0)
        .astype(int)
    )
    reroutes = (
        events_df[events_df["event"] == "reroute"]
        .groupby("t_s")
        .size()
        .reindex(t_index, fill_value=0)
        .astype(int)
    )

    timeseries_df = pd.DataFrame({"t_s": t_index, "installs": installs.values, "reroutes": reroutes.values})

    warmup_s = compute_warmup_seconds(timeseries_df, cfg)

    # t_rel_s para que tus series empiecen en 0 post-warmup
    timeseries_df["t_rel_s"] = (timeseries_df["t_s"] - warmup_s).clip(lower=0).astype(int)
    events_df["t_rel_s"] = (events_df["t_s"] - warmup_s).clip(lower=0).astype(int)

    events_post = events_df[events_df["t_s"] >= warmup_s]

    rep_info = dict(
        controller=controller,
        load=load,
        rep=rep,
        log_file=str(log_path),
        warmup_s=warmup_s,
        t_max_s=tmax,
        installs_total=int((events_df["event"] == "flow_install").sum()),
        installs_post_warmup=int((events_post["event"] == "flow_install").sum()),
        initial_installs_total=int(((events_df["event"] == "flow_install") & (events_df["is_initial"] == True)).sum()),
        initial_installs_post_warmup=int(((events_post["event"] == "flow_install") & (events_post["is_initial"] == True)).sum()),
        reroutes_total=int((events_df["event"] == "reroute").sum()),
        reroutes_post_warmup=int((events_post["event"] == "reroute").sum()),
    )

    return events_df, timeseries_df, rep_info


# -----------------------------
# Aggregation
# -----------------------------

def aggregate_load_summary(rep_summary: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if rep_summary.empty:
        return pd.DataFrame()

    stats_cfg = cfg.get("stats", {}).get("confidence", {})
    ci_enable = bool(stats_cfg.get("enable", True))
    ci_level = float(stats_cfg.get("level", 0.95))

    rows: List[dict] = []
    for (controller, load), g in rep_summary.groupby(["controller", "load"], dropna=False):
        r = {"controller": controller, "load": load, "n_reps": int(len(g))}
        vals_reroutes = [float(x) for x in g["reroutes_post_warmup"].fillna(0).tolist()]
        vals_installs = [float(x) for x in g["installs_post_warmup"].fillna(0).tolist()]
        vals_initial = [float(x) for x in g["initial_installs_post_warmup"].fillna(0).tolist()]
        vals_total = [i + r for i, r in zip(vals_installs, vals_reroutes)]

        metrics_vals = {
            "reroutes_post_warmup": vals_reroutes,
            "installs_post_warmup": vals_installs,
            "initial_installs_post_warmup": vals_initial,
            "total_routes_post_warmup": vals_total,
        }

        for col, vals in metrics_vals.items():
            r[f"{col}_mean"] = float(pd.Series(vals).mean()) if vals else 0.0
            if ci_enable:
                lo, hi = t_confidence_interval(vals, level=ci_level)
                r[f"{col}_ci_lo"] = lo
                r[f"{col}_ci_hi"] = hi
            else:
                r[f"{col}_ci_lo"] = None
                r[f"{col}_ci_hi"] = None
        rows.append(r)

    return pd.DataFrame(rows).sort_values(["controller", "load"]).reset_index(drop=True)


def aggregate_timeseries_mean(all_ts: List[pd.DataFrame], cfg: dict) -> pd.DataFrame:
    if not all_ts:
        return pd.DataFrame()

    min_reps = int(cfg.get("time", {}).get("min_reps_per_t", 2))
    ts = pd.concat(all_ts, ignore_index=True)

    grp = ts.groupby(["controller", "load", "t_rel_s"], dropna=False)
    out = grp.agg(
        reps=("rep", "nunique"),
        installs_mean=("installs", "mean"),
        reroutes_mean=("reroutes", "mean"),
        installs_sum=("installs", "sum"),
        reroutes_sum=("reroutes", "sum"),
    ).reset_index()

    out = out[out["reps"] >= min_reps].copy()
    return out.sort_values(["controller", "load", "t_rel_s"]).reset_index(drop=True)


# -----------------------------
# Plots (opcionales)
# -----------------------------

def plot_bar_by_load(load_summary: pd.DataFrame, plots_dir: Path, cfg: dict) -> None:
    pcfg = cfg.get("plots", {}).get("bar_by_load", {})
    if not pcfg.get("enable", True):
        return

    metric = pcfg.get("metric", "reroutes_post_warmup_mean")
    if load_summary.empty or metric not in load_summary.columns:
        return

    loads_sel = pcfg.get("loads", None)
    if isinstance(loads_sel, str) and loads_sel.lower() == "all":
        loads_sel = None

    labels_cfg = pcfg.get("labels", {}) or {}
    title_tpl = labels_cfg.get("title", "")
    xlabel = labels_cfg.get("xlabel", "")
    ylabel = labels_cfg.get("ylabel", "")
    legend_title = labels_cfg.get("legend", None)
    show_ci = bool(pcfg.get("show_ci", True))

    for controller, g in load_summary.groupby("controller", dropna=False):
        gg = g.copy()
        if loads_sel is not None:
            gg = gg[gg["load"].astype(str).isin([str(x) for x in loads_sel])]
        if gg.empty:
            continue

        gg["__k"] = gg["load"].apply(natural_load_key)
        gg = gg.sort_values("__k")
        x_labels = gg["load"].astype(str).tolist()
        x = list(range(len(gg)))
        y = gg[metric].astype(float).tolist()

        yerr = None
        ci_lo_col = f"{metric.replace('_mean','')}_ci_lo" if metric.endswith("_mean") else f"{metric}_ci_lo"
        ci_hi_col = f"{metric.replace('_mean','')}_ci_hi" if metric.endswith("_mean") else f"{metric}_ci_hi"
        if show_ci and ci_lo_col in gg.columns and ci_hi_col in gg.columns:
            lows = gg[ci_lo_col].astype(float)
            highs = gg[ci_hi_col].astype(float)
            yerr = [y_i - lo_i if pd.notna(lo_i) else 0.0 for y_i, lo_i in zip(y, lows)]
            yerr_hi = [hi_i - y_i if pd.notna(hi_i) else 0.0 for y_i, hi_i in zip(y, highs)]
            yerr = [yerr, yerr_hi]

        fig, ax = plt.subplots()
        ax.bar(x, y, yerr=yerr, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        if title_tpl:
            ax.set_title(title_tpl.format(controller=controller, metric=metric))
        if legend_title:
            ax.legend(title=legend_title)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fig.tight_layout()

        print(f"BAR_STATS controller={controller}:")
        for ld, val in zip(x_labels, y):
            ci_txt = ""
            if show_ci and yerr is not None:
                ci_txt = f" (IC: -{yerr[0][x_labels.index(ld)]:.2f}, +{yerr[1][x_labels.index(ld)]:.2f})"
            print(f"  load={ld} {metric}={val:.3f}{ci_txt}")

        fig.savefig(plots_dir / f"bar_{controller}_{metric}.png", dpi=180)
        plt.close(fig)


def plot_grouped_by_load(load_summary: pd.DataFrame, plots_dir: Path, cfg: dict) -> None:
    pcfg = cfg.get("plots", {}).get("bar_grouped", {})
    if not pcfg.get("enable", False):
        return
    if load_summary.empty:
        return

    metric = pcfg.get("metric", "reroutes_post_warmup_mean")
    if metric not in load_summary.columns:
        return

    controllers_sel = pcfg.get("controllers", "all")
    loads_sel = pcfg.get("loads", None)
    if isinstance(loads_sel, str) and loads_sel.lower() == "all":
        loads_sel = None

    labels_cfg = pcfg.get("labels", {}) or {}
    title_tpl = labels_cfg.get("title", "")
    xlabel = labels_cfg.get("xlabel", "")
    ylabel = labels_cfg.get("ylabel", "")
    legend_title = labels_cfg.get("legend", "Controlador")
    show_ci = bool(pcfg.get("show_ci", True))

    dfp = load_summary.copy()
    if controllers_sel != "all":
        if isinstance(controllers_sel, list):
            dfp = dfp[dfp["controller"].isin(controllers_sel)]
        else:
            dfp = dfp[dfp["controller"] == controllers_sel]
    if loads_sel is not None:
        dfp = dfp[dfp["load"].astype(str).isin([str(x) for x in loads_sel])]
    if dfp.empty:
        return

    loads = sorted(dfp["load"].astype(str).unique().tolist(), key=lambda x: x)
    loads = sorted(loads, key=natural_load_key)
    controllers = sorted(dfp["controller"].astype(str).unique().tolist(), key=lambda x: x)

    width = 0.8
    n_ctrl = max(1, len(controllers))
    bar_w = width / n_ctrl
    x = list(range(len(loads)))

    fig, ax = plt.subplots()
    for j, ctrl in enumerate(controllers):
        sub = dfp[dfp["controller"] == ctrl]
        ys = []
        yerr = [[], []]
        for ld in loads:
            row = sub[sub["load"].astype(str) == ld]
            if row.empty:
                ys.append(float("nan"))
                yerr[0].append(0.0); yerr[1].append(0.0)
                continue
            y = float(row.iloc[0][metric])
            ys.append(y)
            if show_ci:
                ci_lo_col = f"{metric.replace('_mean','')}_ci_lo" if metric.endswith("_mean") else f"{metric}_ci_lo"
                ci_hi_col = f"{metric.replace('_mean','')}_ci_hi" if metric.endswith("_mean") else f"{metric}_ci_hi"
                lo = float(row.iloc[0].get(ci_lo_col, float("nan")))
                hi = float(row.iloc[0].get(ci_hi_col, float("nan")))
                yerr[0].append(max(0.0, y - lo) if not pd.isna(lo) else 0.0)
                yerr[1].append(max(0.0, hi - y) if not pd.isna(hi) else 0.0)
            else:
                yerr[0].append(0.0); yerr[1].append(0.0)
        offsets = [xi - width/2 + (j + 0.5)*bar_w for xi in x]
        ax.bar(offsets, ys, bar_w, yerr=yerr if show_ci else None, capsize=3 if show_ci else 0, label=ctrl)

    ax.set_xticks(x)
    ax.set_xticklabels(loads, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title_tpl:
        ax.set_title(title_tpl.format(metric=metric))
    ax.legend(title=legend_title)
    fig.tight_layout()

    print("BAR_GROUPED_STATS:")
    for ld in loads:
        rows = dfp[dfp["load"].astype(str) == ld]
        vals = rows[["controller", metric]].to_records(index=False)
        print(f"  load={ld}: " + ", ".join([f"{c}={v:.3f}" for c, v in vals]))

    fig.savefig(plots_dir / f"bar_grouped_{metric}.png", dpi=180)
    plt.close(fig)


def plot_timeseries(timeseries_mean: pd.DataFrame, plots_dir: Path, cfg: dict) -> None:
    pcfg = cfg.get("plots", {}).get("timeseries", {})
    if not pcfg.get("enable", True):
        return
    if timeseries_mean.empty:
        return

    select = pcfg.get("select", {})
    sel_load = select.get("load", None)
    metric = pcfg.get("metric", "reroutes_mean")
    loads_sel = pcfg.get("loads", None)
    if isinstance(loads_sel, str) and loads_sel.lower() == "all":
        loads_sel = None

    labels_cfg = pcfg.get("labels", {}) or {}
    title_tpl = labels_cfg.get("title", "")
    xlabel = labels_cfg.get("xlabel", "")
    ylabel = labels_cfg.get("ylabel", "")
    legend_title = labels_cfg.get("legend", None)

    df = timeseries_mean.copy()
    if sel_load is not None and sel_load != "all":
        df = df[df["load"].astype(str) == str(sel_load)]
    if loads_sel is not None:
        df = df[df["load"].astype(str).isin([str(x) for x in loads_sel])]

    for load, g in df.groupby("load", dropna=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for controller, cdf in g.groupby("controller", dropna=False):
            ax.plot(cdf["t_rel_s"], cdf[metric], label=str(controller))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title_tpl:
            ax.set_title(title_tpl.format(metric=metric, load=load))
        if legend_title is not None:
            ax.legend(title=legend_title)
        else:
            ax.legend()
        fig.tight_layout()

        fig.savefig(plots_dir / f"timeseries_{metric}_load_{load}.png", dpi=180)
        plt.close(fig)

def plot_total_routes_grouped(load_summary: pd.DataFrame, plots_dir: Path, cfg: dict) -> None:
    """
    Barras agrupadas: total de rutas instaladas (installs + reroutes) por controlador, para cargas seleccionadas.
    """
    pcfg = cfg.get("plots", {}).get("bar_total_routes", {})
    if not pcfg.get("enable", True):
        return
    if load_summary.empty:
        return

    metric = "total_routes_post_warmup_mean"
    if metric not in load_summary.columns:
        return

    controllers_sel = pcfg.get("controllers", "all")
    loads_sel = pcfg.get("loads", None)
    if isinstance(loads_sel, str) and loads_sel.lower() == "all":
        loads_sel = None

    labels_cfg = pcfg.get("labels", {}) or {}
    title_tpl = labels_cfg.get("title", "Total de rutas instaladas (installs + reroutes)")
    xlabel = labels_cfg.get("xlabel", "Carga")
    ylabel = labels_cfg.get("ylabel", "Total de rutas (prom. por rep)")
    legend_title = labels_cfg.get("legend", "Controlador")
    show_ci = bool(pcfg.get("show_ci", True))
    load_suffix = pcfg.get("load_suffix", None)
    grid = bool(pcfg.get("grid", False))

    dfp = load_summary.copy()
    if controllers_sel != "all":
        if isinstance(controllers_sel, list):
            dfp = dfp[dfp["controller"].isin(controllers_sel)]
        else:
            dfp = dfp[dfp["controller"] == controllers_sel]
    if loads_sel is not None:
        dfp = dfp[dfp["load"].astype(str).isin([str(x) for x in loads_sel])]
    if dfp.empty:
        return

    loads = sorted(dfp["load"].astype(str).unique().tolist(), key=natural_load_key)
    controllers = sorted(dfp["controller"].astype(str).unique().tolist(), key=lambda x: x)

    width = 0.8
    n_ctrl = max(1, len(controllers))
    bar_w = width / n_ctrl
    x = list(range(len(loads)))

    fig, ax = plt.subplots()
    for j, ctrl in enumerate(controllers):
        sub = dfp[dfp["controller"] == ctrl]
        ys = []
        yerr = [[], []]
        for ld in loads:
            row = sub[sub["load"].astype(str) == ld]
            if row.empty:
                ys.append(float("nan"))
                yerr[0].append(0.0); yerr[1].append(0.0)
                continue
            y = float(row.iloc[0][metric])
            ys.append(y)
            if show_ci:
                ci_lo_col = "total_routes_post_warmup_ci_lo"
                ci_hi_col = "total_routes_post_warmup_ci_hi"
                lo = float(row.iloc[0].get(ci_lo_col, float("nan")))
                hi = float(row.iloc[0].get(ci_hi_col, float("nan")))
                yerr[0].append(max(0.0, y - lo) if not pd.isna(lo) else 0.0)
                yerr[1].append(max(0.0, hi - y) if not pd.isna(hi) else 0.0)
            else:
                yerr[0].append(0.0); yerr[1].append(0.0)
        offsets = [xi - width/2 + (j + 0.5)*bar_w for xi in x]
        ax.bar(offsets, ys, bar_w, yerr=yerr if show_ci else None, capsize=3 if show_ci else 0, label=ctrl)

    disp_loads = [format_load_label(ld, load_suffix) for ld in loads]
    ax.set_xticks(x)
    ax.set_xticklabels(disp_loads, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title_tpl:
        ax.set_title(title_tpl)
    ax.legend(title=legend_title)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    print("TABLA: Total de rutas instaladas (installs + reroutes) por carga/controlador (promedio por rep):")
    for ld, label in zip(loads, disp_loads):
        rows = dfp[dfp["load"].astype(str) == ld]
        vals = rows[["controller", metric]].to_records(index=False)
        summary = ", ".join([f"{c}={v:.2f}" for c, v in vals])
        print(f"  load={label}: {summary}")

    fig.savefig(plots_dir / f"bar_total_routes_{metric}.png", dpi=180)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build controller log report (flow installs + reroutes)")
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    if cfg.get("report") != "controller_logs":
        raise SystemExit("Config 'report' debe ser: controller_logs")

    extracted_root = Path(cfg["inputs"]["extracted_root"])
    logs_glob = cfg["inputs"].get("logs_glob", "rep_*/logs/controller*.log")

    controller_filter = cfg["inputs"].get("controller_filter", []) or []
    controller_filter = [str(x) for x in controller_filter]

    out_dir = Path(cfg["outputs"]["out_dir"])
    plots_dir = Path(cfg["outputs"].get("plots_dir", out_dir / "plots"))
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    patterns = build_patterns(cfg)

    all_events: List[pd.DataFrame] = []
    all_ts: List[pd.DataFrame] = []
    rep_rows: List[dict] = []
    missing_rows: List[dict] = []

    if not extracted_root.exists():
        raise SystemExit(f"extracted_root no existe: {extracted_root}")

    controller_dirs = [p for p in extracted_root.iterdir() if p.is_dir()]
    if controller_filter:
        controller_dirs = [p for p in controller_dirs if p.name in controller_filter]

    for cdir in sorted(controller_dirs, key=lambda p: p.name):
        load_dirs = [p for p in cdir.iterdir() if p.is_dir()]
        for ldir in sorted(load_dirs, key=lambda p: p.name):
            log_files = sorted(ldir.glob(logs_glob))
            if not log_files:
                missing_rows.append(
                    dict(controller=cdir.name, load=ldir.name, reason="no log files", expected_glob=logs_glob, dir=str(ldir))
                )
                continue

            for lf in log_files:
                events_df, ts_df, rep_info = parse_log_file(lf, cfg, patterns)
                rep_rows.append(rep_info)

                if not events_df.empty:
                    all_events.append(events_df)

                if not ts_df.empty:
                    controller, load, rep = infer_controller_load_rep(lf)
                    ts_df = ts_df.copy()
                    ts_df["controller"] = controller
                    ts_df["load"] = load
                    ts_df["rep"] = rep
                    all_ts.append(ts_df)

    rep_summary = pd.DataFrame(rep_rows)
    load_summary = aggregate_load_summary(rep_summary, cfg) if not rep_summary.empty else pd.DataFrame()
    timeseries_mean = aggregate_timeseries_mean(all_ts, cfg) if all_ts else pd.DataFrame()

    events_out = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    missing_out = pd.DataFrame(missing_rows)

    csv_cfg = cfg["outputs"]["csv"]
    if "events" in csv_cfg:
        events_out.to_csv(out_dir / csv_cfg["events"], index=False)
    rep_summary.to_csv(out_dir / csv_cfg["rep_summary"], index=False)
    if not load_summary.empty and "load_summary" in csv_cfg:
        load_summary.to_csv(out_dir / csv_cfg["load_summary"], index=False)
    if not timeseries_mean.empty and "timeseries_mean" in csv_cfg:
        timeseries_mean.to_csv(out_dir / csv_cfg["timeseries_mean"], index=False)
    if "missing" in csv_cfg:
        missing_out.to_csv(out_dir / csv_cfg["missing"], index=False)

    # Tabla de promedio de rutas instaladas (installs + reroutes) por controlador/carga (promedio por repetición)
    if not rep_summary.empty:
        rep_summary = rep_summary.copy()
        rep_summary["total_rutas_rep"] = rep_summary["installs_post_warmup"].fillna(0).astype(float) + rep_summary["reroutes_post_warmup"].fillna(0).astype(float)
        tbl = (
            rep_summary.groupby(["controller", "load"], dropna=False)
            .agg(
                installs_mean=("installs_post_warmup", "mean"),
                reroutes_mean=("reroutes_post_warmup", "mean"),
                total_rutas_mean=("total_rutas_rep", "mean"),
                reps=("rep", "nunique"),
            )
            .reset_index()
        )
        tbl["__load_key"] = tbl["load"].apply(natural_load_key)
        tbl = tbl.sort_values(["controller", "__load_key"]).drop(columns=["__load_key"])

        print("TABLA PROMEDIO DE RUTAS INSTALADAS (incluye reroutes) por controlador/carga (promedio por rep):")
        for _, row in tbl.iterrows():
            print(
                f"  controller={row['controller']} load={row['load']} reps={int(row['reps'])}: "
                f"installs_avg={row['installs_mean']:.2f} reroutes_avg={row['reroutes_mean']:.2f} total_avg={row['total_rutas_mean']:.2f}"
            )

    plot_bar_by_load(load_summary, plots_dir, cfg)
    plot_grouped_by_load(load_summary, plots_dir, cfg)
    plot_total_routes_grouped(load_summary, plots_dir, cfg)
    plot_timeseries(timeseries_mean, plots_dir, cfg)

    print("OK")
    print(f"out_dir: {out_dir}")
    print(f"rep_summary rows: {len(rep_summary)}")
    print(f"events rows: {len(events_out)}")
    print(f"load_summary rows: {len(load_summary)}")
    print(f"timeseries_mean rows: {len(timeseries_mean)}")


if __name__ == "__main__":
    main()
