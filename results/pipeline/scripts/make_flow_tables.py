#!/usr/bin/env python3
"""
Genera tablas comparativas por flujo/carga/controlador a partir de load_summary_flow.csv.

Config YAML de ejemplo:
  input_csv: "Results/processed/flows/load_summary_flow.csv"
  output_dir: "Results/processed/flows/tables"
  decimals: 2  # opcional: redondeo global
  flow_labels:  # opcional: alias para mostrar en la tabla
    mqtt_sec_sub: "MQTT"
    rtp_iptv_rec: "IPTV"
  selection:
    controllers: ["Baseline","TE_Agresivo"]   # o "all"
    loads: ["1a","6a","12a"]                  # o "all"
    flows: ["mqtt_sec_sub","rtp_iptv_rec","rtp_voip","udp_noise"]  # o "all"
  tables:
    - name: "delay"
      metric: "delay_mean_ms"   # base del campo en CSV
      stat: "mean"              # mean|min|max
      unit: "ms"
      decimals: 2
      layout: "flow_load_rows"  # (default) filas: flujo|carga, cols: controladores
    - name: "delay_by_load"
      metric: "delay_mean_ms"
      stat: "mean"
      unit: "ms"
      layout: "flow_controller_rows"  # filas: flujo|controlador, cols: cargas
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def pick_ordered(vals, universe: List[str]) -> List[str]:
    if vals is None or vals == "all":
        return list(universe)
    if isinstance(vals, list):
        return [str(x) for x in vals]
    return [str(vals)]


def natural_load_key(s: str):
    """Convierte '10a' -> 10.0 para ordenar, deja otros strings al final."""
    try:
        num = ""
        for ch in str(s):
            if ch.isdigit() or ch == ".":
                num += ch
            else:
                break
        return float(num) if num else float("inf")
    except Exception:
        return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta al YAML de configuración.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    input_csv = Path(cfg["input_csv"]).resolve()
    out_dir = Path(cfg.get("output_dir", input_csv.parent)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise SystemExit(f"CSV vacío: {input_csv}")
    global_decimals = cfg.get("decimals")
    flow_labels = {str(k): str(v) for k, v in (cfg.get("flow_labels") or {}).items()}

    controllers = sorted(df["controller"].dropna().astype(str).unique().tolist())
    loads = sorted(df["load"].dropna().astype(str).unique().tolist())
    flows = sorted(df["id"].dropna().astype(str).unique().tolist())

    sel = cfg.get("selection", {}) or {}
    controllers_sel = pick_ordered(sel.get("controllers"), controllers)
    loads_sel = pick_ordered(sel.get("loads"), loads)
    flows_sel = pick_ordered(sel.get("flows"), flows)
    flow_order = {f: i for i, f in enumerate(flows_sel)}

    df = df[
        df["controller"].astype(str).isin(controllers_sel)
        & df["load"].astype(str).isin(loads_sel)
        & df["id"].astype(str).isin(flows_sel)
    ].copy()

    tables = cfg.get("tables", [])
    if not tables:
        raise SystemExit("Config 'tables' vacío.")

    for tbl in tables:
        metric = tbl["metric"]
        stat = tbl.get("stat", "mean")
        unit = tbl.get("unit", "")
        name = tbl.get("name", metric)
        layout = tbl.get("layout", "flow_load_rows")
        decimals_cfg = tbl.get("decimals", global_decimals)
        decimals = None
        if decimals_cfg is not None:
            try:
                decimals = int(decimals_cfg)
            except Exception:
                print(f"WARNING: 'decimals' invalido ({decimals_cfg}) en tabla {name}; se ignora redondeo")

        col_map = {
            "mean": f"{metric}_mean",
            "min": f"{metric}_min",
            "max": f"{metric}_max",
        }
        col = col_map.get(stat, col_map["mean"])
        if col not in df.columns:
            print(f"WARNING: columna {col} no encontrada; se omite tabla {name}")
            continue

        d = df[["id", "load", "controller", col]].copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        if layout == "flow_controller_rows":
            pivot = d.pivot_table(index=["id", "controller"], columns="load", values=col)
            pivot.columns = [str(c) for c in pivot.columns]
            if not pivot.empty:
                sorted_loads = sorted(pivot.columns, key=natural_load_key)
                pivot = pivot.reset_index()
                pivot = pivot[["id", "controller"] + sorted_loads]
            pivot.columns.name = None
            if not pivot.empty:
                pivot["__flow_order"] = pivot["id"].map(flow_order).fillna(len(flow_order))
                pivot = pivot.sort_values(["__flow_order", "controller"]).drop(columns="__flow_order")
                pivot["id"] = pivot["id"].map(lambda x: flow_labels.get(str(x), x))
        else:  # flow_load_rows (por defecto)
            pivot = d.pivot_table(index=["id", "load"], columns="controller", values=col)
            pivot = pivot.reset_index()
            pivot.columns.name = None
            if not pivot.empty:
                pivot["__load_key"] = pivot["load"].apply(natural_load_key)
                pivot["__flow_order"] = pivot["id"].map(flow_order).fillna(len(flow_order))
                pivot = pivot.sort_values(["__flow_order", "__load_key"]).drop(columns=["__flow_order", "__load_key"])
                pivot["id"] = pivot["id"].map(lambda x: flow_labels.get(str(x), x))

        if decimals is not None and not pivot.empty:
            num_cols = pivot.select_dtypes(include="number").columns
            pivot[num_cols] = pivot[num_cols].round(decimals)

        out_path = out_dir / f"table_{name}.csv"
        pivot.to_csv(out_path, index=False)
        print(f"OK tabla {name} ({stat} {unit}, layout={layout}, decimals={decimals}) -> {out_path}")


if __name__ == "__main__":
    main()
