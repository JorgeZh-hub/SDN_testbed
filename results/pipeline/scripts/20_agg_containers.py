#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import yaml

CONTAINER_COLS = [
    "timestamp","cpu_perc","mem_used_bytes","mem_limit_bytes","mem_perc",
    "net_rx_bytes","net_tx_bytes","block_read_bytes","block_write_bytes","pids"
]

def read_csv_safe(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # normaliza columnas esperadas (si faltan algunas, crea NaN)
    for c in CONTAINER_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def delta_last_first(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return pd.NA
    d = float(s.iloc[-1]) - float(s.iloc[0])
    # si hubo reset del contador, d puede salir negativo
    return d if d >= 0 else pd.NA

def rep_metrics(df: pd.DataFrame) -> dict:
    df = df.sort_values("timestamp")
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    t_start = float(ts.min()) if ts.notna().any() else pd.NA
    t_end   = float(ts.max()) if ts.notna().any() else pd.NA

    cpu = pd.to_numeric(df["cpu_perc"], errors="coerce")
    mem_used = pd.to_numeric(df["mem_used_bytes"], errors="coerce")
    mem_perc = pd.to_numeric(df["mem_perc"], errors="coerce")

    out = {
        "samples": int(df.shape[0]),
        "t_start": t_start,
        "t_end": t_end,
        "duration_s": (t_end - t_start) if (pd.notna(t_start) and pd.notna(t_end)) else pd.NA,

        "cpu_mean_perc": float(cpu.mean(skipna=True)) if cpu.notna().any() else pd.NA,
        "cpu_std_perc": float(cpu.std(skipna=True)) if cpu.notna().any() else pd.NA,

        "mem_mean_bytes": float(mem_used.mean(skipna=True)) if mem_used.notna().any() else pd.NA,
        "mem_max_bytes": float(mem_used.max(skipna=True)) if mem_used.notna().any() else pd.NA,
        "mem_mean_perc": float(mem_perc.mean(skipna=True)) if mem_perc.notna().any() else pd.NA,

        "net_rx_total_bytes": delta_last_first(df["net_rx_bytes"]),
        "net_tx_total_bytes": delta_last_first(df["net_tx_bytes"]),
        "block_read_total_bytes": delta_last_first(df["block_read_bytes"]),
        "block_write_total_bytes": delta_last_first(df["block_write_bytes"]),
    }
    return out

def load_config(cfg_path: Path):
    if not cfg_path.exists():
        return None
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-root", required=True, help="data/extracted")
    ap.add_argument("--out-dir", required=True, help="data/processed/containers")
    ap.add_argument("--config", default="configs/experiments.yml")
    args = ap.parse_args()

    extracted_root = Path(args.extracted_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(args.config))
    container_filter = set((cfg or {}).get("container_filter", []) or [])

    rows = []

    # Esperado: extracted_root/<controller>/<load>/rep_*/reports/containers/*.csv
    for controller_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        controller = controller_dir.name
        for load_dir in sorted([p for p in controller_dir.iterdir() if p.is_dir()]):
            load_label = load_dir.name
            reps = sorted([p for p in load_dir.glob("rep_*") if p.is_dir()])
            for rep_dir in reps:
                rep = rep_dir.name
                cont_dir = rep_dir / "reports" / "containers"
                if not cont_dir.exists():
                    continue

                for csv_path in sorted(cont_dir.glob("*.csv")):
                    container_name = csv_path.stem  # ryu-ctrl, etc.
                    if container_filter and container_name not in container_filter:
                        continue

                    df = read_csv_safe(csv_path)
                    m = rep_metrics(df)

                    row = {
                        "controller": controller,
                        "load": load_label,
                        "rep": rep,
                        "container": container_name,
                        "src_csv": str(csv_path),
                    }
                    row.update(m)
                    rows.append(row)

    rep_df = pd.DataFrame(rows).sort_values(["controller","load","container","rep"])
    rep_out = out_dir / "rep_metrics.csv"
    rep_df.to_csv(rep_out, index=False)

    # Agregación por carga: promedio y desviación sobre reps
    if not rep_df.empty:
        metric_cols = [
            "cpu_mean_perc","mem_mean_bytes","mem_max_bytes","mem_mean_perc",
            "block_read_total_bytes","block_write_total_bytes",
            "net_rx_total_bytes","net_tx_total_bytes",
            "duration_s","samples"
        ]
        grp = rep_df.groupby(["controller","load","container"], dropna=False)
        load_df = grp[metric_cols].agg(["mean","std","min","max","count"]).reset_index()

        # aplana multi-index de columnas
        load_df.columns = [
            c[0] if c[1]=="" else f"{c[0]}_{c[1]}"
            for c in load_df.columns.to_flat_index()
        ]

        load_out = out_dir / "load_metrics.csv"
        load_df.to_csv(load_out, index=False)

    print(f"OK: escrito {rep_out}")
    print(f"OK: escrito {out_dir / 'load_metrics.csv'}")

if __name__ == "__main__":
    main()
