#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import yaml
import matplotlib.pyplot as plt

# networkx es opcional (solo para el snapshot de topología)
try:
    import networkx as nx
except Exception:
    nx = None


# -------------------------
# Helpers (similar al builder de contenedores)
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
    return str(s)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
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
        if abs(level - 0.95) < 1e-6:
            return 1.96
        return 1.96

def compute_ci(mean: float, std: float, n: int, level: float, method: str) -> Tuple[Optional[float], Optional[float]]:
    if n < 2 or mean is None or std is None or pd.isna(mean) or pd.isna(std):
        return (None, None)
    se = std / math.sqrt(n)
    crit = t_critical(level, n - 1) if method == "t" else (1.96 if abs(level - 0.95) < 1e-6 else 1.96)
    return (mean - crit * se, mean + crit * se)

def counter_rate(series: pd.Series, dt_s: float) -> pd.Series:
    """Series de contador acumulado -> tasa (unidades/seg). Resets negativos -> 0."""
    c = to_num(series).ffill()
    d = c.diff()
    d = d.where(d >= 0, pd.NA)
    rate = (d / float(dt_s)).fillna(0.0)
    return rate

def detect_warmup_auto(metric_series: pd.Series, t_series: pd.Series, cfg: dict) -> Optional[int]:
    """
    Busca el primer tiempo donde el "estado" ya arrancó:
      - metric_smooth > th sostenido sustain_s
    """
    th = float(cfg.get("idle_th", 1.0))
    smooth_window_s = int(cfg.get("smooth_window_s", 5))
    sustain_s = int(cfg.get("sustain_s", 5))
    max_search_s = int(cfg.get("max_search_s", 120))

    x = to_num(metric_series).fillna(0.0)

    win = max(1, smooth_window_s)
    x_smooth = x.rolling(window=win, min_periods=1).mean()

    active = (x_smooth > th).astype(int)
    sustain = max(1, sustain_s)
    run = active.rolling(window=sustain, min_periods=sustain).sum()

    mask = (t_series <= max_search_s)
    candidates = run[mask & run.notna() & (run >= sustain)]
    if candidates.empty:
        return None

    first_idx = int(candidates.index[0])
    warmup_start = max(0, int(t_series.iloc[first_idx]) - (sustain - 1))
    return warmup_start


# -------------------------
# Link aggregation + resampling
# -------------------------

def _parse_directed_link_id(link_id: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"^\s*(\d+)\s*->\s*(\d+)", str(link_id))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))

def _parse_pair_link_id(link_id: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)", str(link_id))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))

def _topology_selection(cfg_topo: dict) -> Tuple[Optional[Set[int]], Optional[Set[Tuple[int, int]]]]:
    cfg_topo = cfg_topo or {}
    nodes_raw = cfg_topo.get("nodes", None)
    if nodes_raw is None:
        pos = cfg_topo.get("positions", None)
        if isinstance(pos, dict) and pos:
            nodes_raw = list(pos.keys())

    nodes = set()
    if nodes_raw is not None:
        if isinstance(nodes_raw, dict):
            nodes_iter = nodes_raw.keys()
        elif isinstance(nodes_raw, (list, tuple, set)):
            nodes_iter = nodes_raw
        else:
            nodes_iter = [nodes_raw]
        for n in nodes_iter:
            try:
                nodes.add(int(n))
            except Exception:
                pass

    edges_raw = cfg_topo.get("links", None)
    if edges_raw is None:
        edges_raw = cfg_topo.get("edges", None)
    edges = set()
    if edges_raw:
        for e in edges_raw:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                try:
                    u = int(e[0])
                    v = int(e[1])
                except Exception:
                    continue
                edges.add((min(u, v), max(u, v)))

    nodes_set = nodes if nodes else None
    edges_set = edges if edges else None
    if nodes_set and edges_set:
        edges_set = {pair for pair in edges_set if pair[0] in nodes_set and pair[1] in nodes_set}
    return nodes_set, edges_set

def filter_links_by_topology(df: pd.DataFrame, cfg_topo: dict, id_mode: str) -> pd.DataFrame:
    nodes_set, edges_set = _topology_selection(cfg_topo)
    if nodes_set is None and edges_set is None:
        return df

    def _keep(link_id: str) -> bool:
        if id_mode in ("directed", "directed_port"):
            pair = _parse_directed_link_id(link_id)
        else:
            pair = _parse_pair_link_id(link_id)
        if pair is None:
            return False
        u, v = pair
        if nodes_set and (u not in nodes_set or v not in nodes_set):
            return False
        if edges_set and (min(u, v), max(u, v)) not in edges_set:
            return False
        return True

    mask = df["link_id"].map(_keep)
    return df[mask]

def order_link_ids_for_heatmap(link_ids: List[str], id_mode: str, order_mode: str = "minmax") -> List[str]:
    seen = set()
    unique: List[str] = []
    for lid in link_ids:
        if pd.isna(lid):
            continue
        lid = str(lid)
        if lid in seen:
            continue
        seen.add(lid)
        unique.append(lid)

    if id_mode not in ("directed", "directed_port"):
        return sorted(unique)

    items = []
    leftovers = []
    for lid in unique:
        pair = _parse_directed_link_id(lid)
        if pair is None:
            leftovers.append(lid)
            continue
        src, dst = pair
        a, b = min(src, dst), max(src, dst)
        if order_mode == "maxmin":
            dir_key = 0 if src == b else 1
        else:
            dir_key = 0 if src == a else 1
        items.append((a, b, dir_key, lid))

    items.sort()
    ordered = [lid for _, _, _, lid in items]
    ordered.extend(sorted(leftovers))
    return ordered

def build_link_id(df: pd.DataFrame, mode: str) -> pd.Series:
    """
    mode:
      - directed: "src->dst"
      - directed_port: "src->dst:p<port>"
      - undirected_pair: "min-max"
      - undirected_pair_port: "min-max:p<port>"  (útil si hay enlaces paralelos)
    """
    src = to_num(df["src_dpid"]).astype("Int64")
    dst = to_num(df["dst_dpid"]).astype("Int64")
    port = to_num(df["port_no"]).astype("Int64")

    if mode == "directed":
        return src.astype(str) + "->" + dst.astype(str)

    if mode == "directed_port":
        return src.astype(str) + "->" + dst.astype(str) + ":p" + port.astype(str)

    a = src.where(src <= dst, dst)
    b = dst.where(src <= dst, src)

    if mode == "undirected_pair":
        return a.astype(str) + "-" + b.astype(str)

    if mode == "undirected_pair_port":
        return a.astype(str) + "-" + b.astype(str) + ":p" + port.astype(str)

    # fallback
    return a.astype(str) + "-" + b.astype(str)

def aggregate_links_per_timestamp(df: pd.DataFrame, cfg_links: dict) -> pd.DataFrame:
    """
    A partir de registros por puerto, agregamos por link_id en cada timestamp.
    """
    id_mode = cfg_links.get("id_mode", "undirected_pair")
    util_agg = cfg_links.get("agg", {}).get("util", "max")
    mbps_agg = cfg_links.get("agg", {}).get("mbps", "sum")
    drops_agg = cfg_links.get("agg", {}).get("drops", "sum")

    df2 = df.copy()
    df2["timestamp"] = to_num(df2["timestamp"])
    df2 = df2.dropna(subset=["timestamp"])

    df2["link_id"] = build_link_id(df2, mode=id_mode)

    # normaliza columnas numéricas
    for c in ["mbps", "rx_drop", "tx_drop", "total_drop", "util_pct"]:
        df2[c] = to_num(df2.get(c, pd.Series(dtype=float)))

    g = df2.groupby(["timestamp", "link_id"], dropna=False)

    # util_pct: max o mean
    if util_agg == "mean":
        util = g["util_pct"].mean()
    else:
        util = g["util_pct"].max()

    # mbps: sum o mean o max
    if mbps_agg == "mean":
        mbps = g["mbps"].mean()
    elif mbps_agg == "max":
        mbps = g["mbps"].max()
    else:
        mbps = g["mbps"].sum()

    # drops: sum (normalmente) o max
    if drops_agg == "max":
        drops = g["total_drop"].max()
    else:
        drops = g["total_drop"].sum()

    out = pd.concat([util.rename("util_pct"), mbps.rename("mbps"), drops.rename("total_drop")], axis=1).reset_index()
    return out

def resample_link_timeseries(df_link: pd.DataFrame, resample_s: int) -> pd.DataFrame:
    """
    df_link: columnas [timestamp, link_id, util_pct, mbps, total_drop]
    Retorna: [link_id, t_rel_s, util_pct, mbps, total_drop, drop_rate]
    (util_pct/mbps: sample-and-hold; total_drop: last + ffill; drop_rate: diff/dt)
    """
    if df_link.empty:
        return pd.DataFrame()

    df = df_link.copy()
    df["timestamp"] = to_num(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values(["link_id", "timestamp"])

    t0 = float(df["timestamp"].min())
    df["t_rel_s"] = (df["timestamp"] - t0).astype(int)

    # resample por link_id
    freq = f"{int(resample_s)}S"
    out_rows = []

    for link_id, g in df.groupby("link_id", dropna=False):
        gg = g.copy()
        gg = gg.sort_values("t_rel_s")
        idx = pd.to_timedelta(gg["t_rel_s"], unit="s")
        gg.index = idx

        util = gg[["util_pct"]].resample(freq).last().ffill()
        mbps = gg[["mbps"]].resample(freq).last().ffill()
        drops = gg[["total_drop"]].resample(freq).last().ffill()

        out = pd.concat([util, mbps, drops], axis=1)
        out["t_rel_s"] = out.index.total_seconds().astype(int)
        out["link_id"] = str(link_id)

        out["drop_rate"] = counter_rate(out["total_drop"], dt_s=resample_s)

        out_rows.append(out.reset_index(drop=True))

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

def compute_global_timeseries(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    De link-timeseries (por link_id) crea series globales por t_rel_s:
      mean_util, p95_util, max_util, total_mbps, sum_drop_rate
    """
    if df_res.empty:
        return pd.DataFrame()

    g = df_res.groupby("t_rel_s", dropna=False)

    def p95(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(x.quantile(0.95)) if len(x) else None

    out = pd.DataFrame({
        "t_rel_s": g.size().index.astype(int),
        "links_n": g.size().values.astype(int),
        "mean_util": g["util_pct"].mean().values,
        "p95_util": g["util_pct"].apply(p95).values,
        "max_util": g["util_pct"].max().values,
        "total_mbps": g["mbps"].sum().values,
        "sum_drop_rate": g["drop_rate"].sum().values,
    })
    return out.sort_values("t_rel_s")


# -------------------------
# Rep summary
# -------------------------

def compute_rep_summary_links(df_raw: pd.DataFrame, cfg: dict) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - metrics_out (post-warmup)
      - meta (warmup_s, t0, t_peak, etc.)
      - df_links_res (link-level resampled)
      - df_global (global timeseries)
    """
    required = cfg["inputs"].get("required_columns", [])
    df = ensure_columns(df_raw.copy(), required)

    # Agrega por link y timestamp
    df_agg = aggregate_links_per_timestamp(df, cfg_links=cfg.get("links", {}))

    resample_s = int(cfg["time"].get("resample_s", 1))
    df_res = resample_link_timeseries(df_agg, resample_s=resample_s)

    df_global = compute_global_timeseries(df_res)

    if df_res.empty or df_global.empty:
        return {}, {"error": "empty_after_resample"}, pd.DataFrame(), pd.DataFrame()

    # Warmup
    warm_cfg = cfg.get("warmup", {"mode": "none"})
    mode = warm_cfg.get("mode", "none")
    warmup_s = 0
    if mode == "none":
        warmup_s = 0
    elif mode == "fixed":
        warmup_s = int(warm_cfg.get("fixed", {}).get("seconds", 0))
    elif mode == "auto":
        auto = warm_cfg.get("auto", {})
        metric_name = auto.get("metric", "mean_util")
        series = df_global.get(metric_name, pd.Series(dtype=float))
        w = detect_warmup_auto(series, df_global["t_rel_s"], auto)
        warmup_s = int(warm_cfg.get("fallback_if_not_found", 0)) if w is None else int(w)
    else:
        warmup_s = 0

    # Segmento post-warmup
    glob_seg = df_global[df_global["t_rel_s"] >= warmup_s].copy()
    link_seg = df_res[df_res["t_rel_s"] >= warmup_s].copy()

    # Pico (peor momento) definido por métrica global
    peak_cfg = cfg.get("peak", {})
    peak_metric = peak_cfg.get("metric", "mean_util")
    if glob_seg.empty or peak_metric not in glob_seg.columns:
        t_peak = warmup_s
    else:
        t_peak = int(glob_seg.loc[glob_seg[peak_metric].idxmax(), "t_rel_s"])

    meta = {
        "t0_epoch": float(to_num(df_raw["timestamp"]).dropna().min()),
        "warmup_s": warmup_s,
        "t_peak_s": t_peak,
        "duration_total_s": int(df_global["t_rel_s"].max()),
        "duration_used_s": int(glob_seg["t_rel_s"].max() - glob_seg["t_rel_s"].min()) if len(glob_seg) >= 2 else 0,
    }

    # Resúmenes (post-warmup)
    metrics_out = {
        "mean_util_avg": float(glob_seg["mean_util"].mean()) if glob_seg["mean_util"].notna().any() else None,
        "p95_util_avg": float(glob_seg["p95_util"].mean()) if glob_seg["p95_util"].notna().any() else None,
        "max_util_peak": float(glob_seg["max_util"].max()) if glob_seg["max_util"].notna().any() else None,
        "total_mbps_avg": float(glob_seg["total_mbps"].mean()) if glob_seg["total_mbps"].notna().any() else None,
        "sum_drop_rate_avg": float(glob_seg["sum_drop_rate"].mean()) if glob_seg["sum_drop_rate"].notna().any() else None,
        "drops_total": float(link_seg.groupby("link_id")["total_drop"].apply(lambda s: (to_num(s).iloc[-1] - to_num(s).iloc[0]) if len(to_num(s).dropna())>=2 and (to_num(s).iloc[-1] - to_num(s).iloc[0])>=0 else 0.0).sum()) if not link_seg.empty else None,
    }

    return metrics_out, meta, df_res, df_global


# -------------------------
# Plotting
# -------------------------

def plot_heatmap(
    df_link_mean: pd.DataFrame,
    out_path: Path,
    value_col: str,
    title: str,
    vmax: Optional[float] = None,
    index_order: Optional[List[str]] = None,
):
    """
    df_link_mean: columnas [link_id, t_rel_s, <value_col>]
    """
    if df_link_mean.empty:
        print(f"WARNING: heatmap sin datos -> {out_path}")
        return

    pivot = df_link_mean.pivot(index="link_id", columns="t_rel_s", values=value_col)
    pivot = pivot.sort_index()
    if index_order:
        ordered = [lid for lid in index_order if lid in pivot.index]
        if len(ordered) < len(pivot.index):
            rest = [lid for lid in pivot.index if lid not in ordered]
            ordered.extend(rest)
        pivot = pivot.reindex(ordered)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.25 * len(pivot.index))))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest", vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel("link_id")

    # ticks: no pongas todos si hay muchos
    xticks = list(range(0, pivot.shape[1], max(1, pivot.shape[1] // 10)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(pivot.columns[i]) for i in xticks], rotation=0)

    yticks = list(range(pivot.shape[0]))
    ax.set_yticks(yticks)
    ax.set_yticklabels(pivot.index.tolist())

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _parse_positions(cfg_topo: dict) -> Dict[int, Tuple[float, float]]:
    """
    Soporta:
      topology:
        positions:
          "1": [0, 0]
          "2": [1, 0]
    Retorna {dpid_int: (x,y)}
    """
    pos = {}
    raw = (cfg_topo or {}).get("positions", {}) or {}
    for k, v in raw.items():
        try:
            dpid = int(k)
        except Exception:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            pos[dpid] = (float(v[0]), float(v[1]))
    return pos

def _parse_edges(cfg_topo: dict) -> List[Tuple[int, int, dict]]:
    """
    Soporta:
      topology:
        links:
          - [1, 2, {"capacity_mbps": 90}]
          - [2, 3]
    """
    edges = []
    cfg_topo = cfg_topo or {}
    raw_edges = cfg_topo.get("links", None)
    if raw_edges is None:
        raw_edges = cfg_topo.get("edges", []) or []
    for e in raw_edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            u = int(e[0]); v = int(e[1])
            attrs = {}
            if len(e) >= 3 and isinstance(e[2], dict):
                attrs = dict(e[2])
            edges.append((u, v, attrs))
    return edges

def plot_topology_snapshot(
    df_link_mean: pd.DataFrame,
    cfg_topo: dict,
    t_peak_s: int,
    out_path: Path,
    title: str,
    edge_agg: str = "max",
):
    if nx is None:
        print("WARNING: networkx no está instalado; se omite snapshot de topología.")
        return

    pos = _parse_positions(cfg_topo)
    edges = _parse_edges(cfg_topo)
    nodes_sel, edges_sel = _topology_selection(cfg_topo)
    if nodes_sel:
        edges = [e for e in edges if e[0] in nodes_sel and e[1] in nodes_sel]
    if edges_sel:
        edges = [e for e in edges if (min(e[0], e[1]), max(e[0], e[1])) in edges_sel]

    if not edges:
        print(f"WARNING: topology.links vacío; no se puede dibujar grafo -> {out_path}")
        return

    # extrae valores en t_peak
    dft = df_link_mean[df_link_mean["t_rel_s"] == int(t_peak_s)].copy()
    # construye mapa util por par (u-v)
    util_map: Dict[Tuple[int,int], float] = {}
    drop_map: Dict[Tuple[int,int], float] = {}

    def _pair_from_link_id(link_id: str) -> Optional[Tuple[int,int]]:
        # link_id tipo "1-2", "1-2:p3", "1->2" o "1->2:p3"
        pair = _parse_directed_link_id(link_id)
        if pair is None:
            pair = _parse_pair_link_id(link_id)
        if pair is None:
            return None
        a, b = pair
        return (min(a, b), max(a, b))

    for _, r in dft.iterrows():
        pair = _pair_from_link_id(r.get("link_id"))
        if pair is None:
            continue
        util = float(r.get("util_pct_mean", 0.0)) if pd.notna(r.get("util_pct_mean", pd.NA)) else 0.0
        dr   = float(r.get("drop_rate_mean", 0.0)) if pd.notna(r.get("drop_rate_mean", pd.NA)) else 0.0

        if pair not in util_map:
            util_map[pair] = util
            drop_map[pair] = dr
        else:
            if edge_agg == "mean":
                util_map[pair] = 0.5 * (util_map[pair] + util)
                drop_map[pair] = 0.5 * (drop_map[pair] + dr)
            else:
                util_map[pair] = max(util_map[pair], util)
                drop_map[pair] = max(drop_map[pair], dr)

    # arma grafo
    G = nx.Graph()
    for u, v, attrs in edges:
        G.add_edge(u, v, **attrs)

    # posiciones: si faltan nodos, layout automático pero estable
    if len(pos) < len(G.nodes()):
        # fallback: spring_layout
        auto = nx.spring_layout(G, seed=1)
        for n in G.nodes():
            if n not in pos:
                pos[n] = (float(auto[n][0]), float(auto[n][1]))

    # estilos por utilización
    utils = []
    widths = []
    for (u, v) in G.edges():
        pair = (min(int(u), int(v)), max(int(u), int(v)))
        util = float(util_map.get(pair, 0.0))
        utils.append(util)
        widths.append(1.0 + 4.0 * (util / 100.0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)

    # Colormap para util
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0.0, vmax=100.0)
    cmap = plt.cm.viridis
    edge_colors = [cmap(norm(u)) for u in utils]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: f"S{n}" for n in G.nodes()}, font_size=10)
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=edge_colors)

    # labels de arista: util% y (opcional) capacidad
    edge_labels = {}
    for (u, v, attrs) in G.edges(data=True):
        pair = (min(int(u), int(v)), max(int(u), int(v)))
        util = util_map.get(pair, 0.0)
        cap = attrs.get("capacity_mbps", None)
        if cap is not None:
            edge_labels[(u, v)] = f"{util:.1f}%"#  / {cap} Mbps"
        else:
            edge_labels[(u, v)] = f"{util:.1f}%"
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("util_pct (%)")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reports_links.yml")
    ap.add_argument("--regenerate-csvs", action="store_true", help="Recalcula y reescribe CSVs aunque ya existan.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    extracted_root = Path(cfg["inputs"]["extracted_root"]).resolve()
    glob_pat = cfg["inputs"]["links_glob"]
    required_cols = cfg["inputs"].get("required_columns", []) or []

    resample_s = int(cfg["time"].get("resample_s", 1))
    min_reps_per_t = int(cfg["time"].get("min_reps_per_t", 1))

    out_dir = Path(cfg["outputs"]["out_dir"]).resolve()
    plots_dir = Path(cfg["outputs"]["plots_dir"]).resolve()
    safe_mkdir(out_dir); safe_mkdir(plots_dir)

    rep_summary_path = out_dir / cfg["outputs"]["csv"]["rep_summary"]
    load_summary_path = out_dir / cfg["outputs"]["csv"]["load_summary"]
    global_ts_path = out_dir / cfg["outputs"]["csv"]["global_timeseries_mean"]
    link_ts_path = out_dir / cfg["outputs"]["csv"]["link_timeseries_mean"]
    missing_path = out_dir / cfg["outputs"]["csv"]["missing"]

    required_csvs = [rep_summary_path, load_summary_path, global_ts_path, link_ts_path]
    use_existing_csvs = (not args.regenerate_csvs and all(p.exists() for p in required_csvs))
    if use_existing_csvs:
        print("INFO: usando CSVs existentes (usa --regenerate-csvs para regenerarlos).")
        rep_df = read_csv_maybe_empty(rep_summary_path)
        load_df = read_csv_maybe_empty(load_summary_path)
        global_mean_df = read_csv_maybe_empty(global_ts_path)
        link_mean_df = read_csv_maybe_empty(link_ts_path)
        miss_df = read_csv_maybe_empty(missing_path)
    else:
        missing_rows = []
        rep_rows = []
        global_ts_rows = []     # por rep (para luego agregar sobre reps)
        link_ts_rows = []       # por rep (para luego agregar sobre reps, por link)
    
        if not extracted_root.exists():
            print(f"ERROR: extracted_root no existe: {extracted_root}")
            return
    
        controllers = sorted([p for p in extracted_root.iterdir() if p.is_dir()])
        if not controllers:
            print("WARNING: no se encontraron controladores.")
            return
    
        for ctrl_dir in controllers:
            controller = ctrl_dir.name
            loads = sorted([p for p in ctrl_dir.iterdir() if p.is_dir()], key=lambda p: natural_load_key(p.name))
    
            for load_dir in loads:
                load = load_dir.name
    
                csvs = list(load_dir.glob(glob_pat))
                if not csvs:
                    missing_rows.append({"controller": controller, "load": load, "rep": None, "path": str(load_dir), "reason": "no_csvs_found"})
                    continue
    
                for csv_path in sorted(csvs):
                    mrep = re.search(r"(rep_\d+)", str(csv_path))
                    rep = mrep.group(1) if mrep else "rep_unknown"
    
                    try:
                        df_raw = pd.read_csv(csv_path)
                    except Exception as e:
                        missing_rows.append({"controller": controller, "load": load, "rep": rep, "path": str(csv_path), "reason": f"read_error:{e}"})
                        continue
    
                    if df_raw.empty:
                        missing_rows.append({"controller": controller, "load": load, "rep": rep, "path": str(csv_path), "reason": "empty_csv"})
                        continue
    
                    # compute rep summary + series
                    metrics_out, meta, df_links_res, df_global = compute_rep_summary_links(df_raw, cfg)
                    if "error" in meta:
                        missing_rows.append({"controller": controller, "load": load, "rep": rep, "path": str(csv_path), "reason": meta["error"]})
                        continue
    
                    rep_rows.append({
                        "controller": controller,
                        "load": load,
                        "rep": rep,
                        "src_csv": str(csv_path),
                        **meta,
                        **metrics_out,
                    })
    
                    # guarda series por rep
                    for _, r in df_global.iterrows():
                        global_ts_rows.append({
                            "controller": controller,
                            "load": load,
                            "rep": rep,
                            "t_rel_s": int(r["t_rel_s"]),
                            "links_n": int(r["links_n"]),
                            "mean_util": float(r["mean_util"]) if pd.notna(r["mean_util"]) else None,
                            "p95_util": float(r["p95_util"]) if pd.notna(r["p95_util"]) else None,
                            "max_util": float(r["max_util"]) if pd.notna(r["max_util"]) else None,
                            "total_mbps": float(r["total_mbps"]) if pd.notna(r["total_mbps"]) else None,
                            "sum_drop_rate": float(r["sum_drop_rate"]) if pd.notna(r["sum_drop_rate"]) else None,
                        })
    
                    for _, r in df_links_res.iterrows():
                        link_ts_rows.append({
                            "controller": controller,
                            "load": load,
                            "rep": rep,
                            "link_id": r["link_id"],
                            "t_rel_s": int(r["t_rel_s"]),
                            "util_pct": float(r["util_pct"]) if pd.notna(r["util_pct"]) else None,
                            "mbps": float(r["mbps"]) if pd.notna(r["mbps"]) else None,
                            "drop_rate": float(r["drop_rate"]) if pd.notna(r["drop_rate"]) else None,
                        })
    
        rep_df = pd.DataFrame(rep_rows)
        miss_df = pd.DataFrame(missing_rows)
        global_rep_df = pd.DataFrame(global_ts_rows)
        link_rep_df = pd.DataFrame(link_ts_rows)
    
        rep_df.to_csv(rep_summary_path, index=False)
        miss_df.to_csv(missing_path, index=False)
    
        # ----------------- load_summary (agrega sobre reps) -----------------
        conf = cfg.get("stats", {}).get("confidence", {})
        ci_enable = bool(conf.get("enable", True))
        ci_level = float(conf.get("level", 0.95))
        ci_method = conf.get("method", "t")
    
        metric_names = cfg.get("metrics", {}).get("summary_metrics", [
            "mean_util_avg", "p95_util_avg", "max_util_peak", "total_mbps_avg", "sum_drop_rate_avg", "drops_total"
        ])
    
        load_rows = []
        if not rep_df.empty:
            grp = rep_df.groupby(["controller", "load"], dropna=False)
            for (controller, load), g in grp:
                out = {"controller": controller, "load": load}
                for m in metric_names:
                    v = pd.to_numeric(g.get(m, pd.Series(dtype=float)), errors="coerce").dropna()
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
            load_df["__load_key"] = load_df["load"].apply(natural_load_key)
            load_df = load_df.sort_values(["controller", "__load_key"]).drop(columns=["__load_key"])
        load_df.to_csv(load_summary_path, index=False)
    
        # ----------------- global_timeseries_mean (agrega sobre reps por t) -----------------
        global_rows_out = []
        if not global_rep_df.empty:
            g2 = global_rep_df.groupby(["controller", "load", "t_rel_s"], dropna=False)
            for (controller, load, t_rel_s), g in g2:
                out = {"controller": controller, "load": load, "t_rel_s": int(t_rel_s)}
                for col in ["mean_util", "p95_util", "max_util", "total_mbps", "sum_drop_rate"]:
                    v = pd.to_numeric(g[col], errors="coerce").dropna()
                    n = int(v.shape[0])
                    mean = float(v.mean()) if n else None
                    std = float(v.std(ddof=1)) if n >= 2 else None
                    out[f"{col}_mean"] = mean
                    out[f"{col}_std"] = std
                    out[f"{col}_count"] = n
                    if ci_enable and (n >= 2) and (mean is not None) and (std is not None):
                        lo, hi = compute_ci(mean, std, n, ci_level, ci_method)
                        out[f"{col}_ci95_low"] = lo
                        out[f"{col}_ci95_high"] = hi
                    else:
                        out[f"{col}_ci95_low"] = None
                        out[f"{col}_ci95_high"] = None
                global_rows_out.append(out)
    
        global_mean_df = pd.DataFrame(global_rows_out)
        if not global_mean_df.empty:
            global_mean_df["__load_key"] = global_mean_df["load"].apply(natural_load_key)
            global_mean_df = global_mean_df.sort_values(["controller", "__load_key", "t_rel_s"]).drop(columns=["__load_key"])
        global_mean_df.to_csv(global_ts_path, index=False)
    
        # ----------------- link_timeseries_mean (agrega sobre reps por link/t) -----------------
        link_rows_out = []
        if not link_rep_df.empty:
            g3 = link_rep_df.groupby(["controller", "load", "link_id", "t_rel_s"], dropna=False)
            for (controller, load, link_id, t_rel_s), g in g3:
                out = {"controller": controller, "load": load, "link_id": link_id, "t_rel_s": int(t_rel_s)}
                for col in ["util_pct", "mbps", "drop_rate"]:
                    v = pd.to_numeric(g[col], errors="coerce").dropna()
                    n = int(v.shape[0])
                    mean = float(v.mean()) if n else None
                    out[f"{col}_mean"] = mean
                    out[f"{col}_count"] = n
                link_rows_out.append(out)
    
        link_mean_df = pd.DataFrame(link_rows_out)
        if not link_mean_df.empty:
            link_mean_df["__load_key"] = link_mean_df["load"].apply(natural_load_key)
            link_mean_df = link_mean_df.sort_values(["controller", "__load_key", "link_id", "t_rel_s"]).drop(columns=["__load_key"])
        link_mean_df.to_csv(link_ts_path, index=False)
    
        print(f"OK: {rep_summary_path}")
        print(f"OK: {load_summary_path}")
        print(f"OK: {global_ts_path}")
        print(f"OK: {link_ts_path}")
        print(f"OK: {missing_path}")
    
    # ----------------- Plots -----------------
    plots_cfg = cfg.get("plots", {})

    # Heatmaps (util y drops) para una selección
    heat_cfg = plots_cfg.get("heatmaps", {})
    if heat_cfg.get("enable", False) and not link_mean_df.empty:
        sel = heat_cfg.get("select", {})
        dfp = link_mean_df.copy()
        if sel.get("controller") is not None:
            dfp = dfp[dfp["controller"] == sel["controller"]]
        if sel.get("load") is not None:
            dfp = dfp[dfp["load"] == sel["load"]]
        topo_cfg = cfg.get("topology", {})
        id_mode = cfg.get("links", {}).get("id_mode", "undirected_pair")
        dfp = filter_links_by_topology(dfp, topo_cfg, id_mode)

        # recorta warmup si el usuario lo pide y hay rep_summary para ese escenario
        warmup_mode = heat_cfg.get("apply_warmup_cut", True)
        warmup_s = 0
        if warmup_mode and not rep_df.empty and sel.get("controller") and sel.get("load"):
            rr = rep_df[(rep_df["controller"] == sel["controller"]) & (rep_df["load"] == sel["load"])]
            if not rr.empty:
                warmup_s = int(pd.to_numeric(rr["warmup_s"], errors="coerce").median(skipna=True))

        if warmup_s > 0:
            dfp = dfp[dfp["t_rel_s"] >= warmup_s]

        # filtra tiempos con suficientes reps (solo visual)
        min_reps = int(heat_cfg.get("min_reps_per_t", 1))
        dfp = dfp[dfp["util_pct_count"] >= min_reps]

        order_mode = heat_cfg.get("directed_pair_order", "minmax")
        index_order = order_link_ids_for_heatmap(dfp["link_id"].tolist(), id_mode, order_mode=order_mode)

        out_u = plots_dir / f"heat_util_{sel.get('controller','all')}_{sel.get('load','all')}.png"
        plot_heatmap(
            dfp,
            out_u,
            value_col="util_pct_mean",
            title=f"Heatmap util_pct (ctrl={sel.get('controller')}, load={sel.get('load')})",
            vmax=100.0,
            index_order=index_order,
        )
        print(f"OK plot: {out_u}")

        out_d = plots_dir / f"heat_drop_{sel.get('controller','all')}_{sel.get('load','all')}.png"
        plot_heatmap(
            dfp,
            out_d,
            value_col="drop_rate_mean",
            title=f"Heatmap drop_rate (drops/s) (ctrl={sel.get('controller')}, load={sel.get('load')})",
            vmax=None,
            index_order=index_order,
        )
        print(f"OK plot: {out_d}")

    # Snapshot topología en el peor instante (por carga) para UN controlador
    snap_cfg = plots_cfg.get("topology_snapshot_peak", {})
    if snap_cfg.get("enable", False) and not rep_df.empty and not link_mean_df.empty:
        if nx is None:
            print("WARNING: networkx no disponible; omito topology_snapshot_peak.")
        else:
            ctrl = snap_cfg.get("controller", None)
            if ctrl is None:
                # fallback: primer controlador
                ctrl = sorted(rep_df["controller"].dropna().unique().tolist())[0]

            loads_sel = snap_cfg.get("loads", "all")
            loads_list = sorted(rep_df[rep_df["controller"] == ctrl]["load"].dropna().unique().tolist(), key=natural_load_key)
            if isinstance(loads_sel, list):
                loads_list = [x for x in loads_list if x in loads_sel]

            topo_cfg = cfg.get("topology", {})
            edge_agg = snap_cfg.get("edge_agg", "max")

            for ld in loads_list:
                rr = rep_df[(rep_df["controller"] == ctrl) & (rep_df["load"] == ld)]
                if rr.empty:
                    continue
                # usa el t_peak mediano entre reps para robustez
                t_peak = int(pd.to_numeric(rr["t_peak_s"], errors="coerce").median(skipna=True))
                warmup_s = int(pd.to_numeric(rr["warmup_s"], errors="coerce").median(skipna=True))

                dfp = link_mean_df[(link_mean_df["controller"] == ctrl) & (link_mean_df["load"] == ld)].copy()
                # recorta warmup para elegir valores consistentes
                dfp = dfp[dfp["t_rel_s"] >= warmup_s]

                outp = plots_dir / f"topo_peak_{ctrl}_{ld}_t{t_peak}.png"
                title = f"Topología @ pico (ctrl={ctrl}, load={ld}, t={t_peak}s)"
                plot_topology_snapshot(
                    dfp.rename(columns={
                        "util_pct_mean":"util_pct_mean",
                        "drop_rate_mean":"drop_rate_mean",
                    }),
                    cfg_topo=topo_cfg,
                    t_peak_s=t_peak,
                    out_path=outp,
                    title=title,
                    edge_agg=edge_agg,
                )
                print(f"OK plot: {outp}")

if __name__ == "__main__":
    main()
