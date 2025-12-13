#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_CACHE = {}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_flow_csv(results_dir, flow_id):
    csv_path = Path(results_dir) / f"{flow_id}.csv"
    cache_key = csv_path.resolve()
    if cache_key in CSV_CACHE:
        return CSV_CACHE[cache_key]

    if not csv_path.exists():
        print(f"   ⚠️  No se encontró {csv_path}, se omite.")
        CSV_CACHE[cache_key] = None
        return None

    df = pd.read_csv(csv_path)
    CSV_CACHE[cache_key] = df
    return df


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_all_flows(cfg):
    flows = {}
    for sc in cfg.get("scenarios", []):
        for f in sc.get("flows", []):
            flows.setdefault(f["id"], f.get("label", f["id"]))
    return flows


def build_flow_colors(flow_ids):
    cmap = plt.get_cmap("tab20")
    return {fid: cmap(i % cmap.N) for i, fid in enumerate(flow_ids)}


def determine_throughput_scale(cfg):
    max_bps = 0.0
    for sc in cfg.get("scenarios", []):
        res_dir = sc["results_dir"]
        for f in sc.get("flows", []):
            df = load_flow_csv(res_dir, f["id"])
            if df is None or "throughput_bytes" not in df.columns:
                continue
            max_bps = max(max_bps, df["throughput_bytes"].max() * 8)

    if max_bps >= 1e6:
        return 1e6, "Mbps"
    return 1e3, "Kbps"


def normalize_flow_filter(flow_filter):
    if not flow_filter:
        return None, {}
    flow_ids = []
    labels = {}
    for item in flow_filter:
        if isinstance(item, dict):
            fid = item.get("id")
            if not fid:
                continue
            flow_ids.append(fid)
            if "label" in item:
                labels[fid] = item["label"]
        else:
            flow_ids.append(str(item))
    return flow_ids, labels


def expand_plot_configs(plot_cfg):
    if isinstance(plot_cfg, list):
        return plot_cfg
    if isinstance(plot_cfg, dict) and plot_cfg.get("groups"):
        base = {k: v for k, v in plot_cfg.items() if k != "groups"}
        return [{**base, **group} for group in plot_cfg["groups"]]
    return [plot_cfg]


# ---------------- PLOTS ----------------

def plot_grouped_boxplot(
    cfg,
    output_dir,
    column,
    title,
    ylabel,
    filename,
    transform=None,
    show_anomalies=True,
    flow_filter=None,
    flow_labels_override=None,
):
    scenarios = cfg["scenarios"]
    all_flows = collect_all_flows(cfg)
    merged_labels = {**all_flows, **(flow_labels_override or {})}

    flow_ids = list(all_flows.keys())
    if flow_filter:
        allowed = set(flow_filter)
        flow_ids = [fid for fid in flow_ids if fid in allowed]
    if not flow_ids:
        print(f"   ⚠️  No hay flujos para {title}.")
        return

    flow_colors = build_flow_colors(flow_ids)
    base_positions = np.arange(len(scenarios))
    group_width = 0.8
    offsets = np.linspace(-group_width / 2, group_width / 2, len(flow_ids))

    data = []
    positions = []
    colors = []
    flows_with_data = set()

    for i, sc in enumerate(scenarios):
        res_dir = sc["results_dir"]
        for j, flow_id in enumerate(flow_ids):
            df = load_flow_csv(res_dir, flow_id)
            if df is None or column not in df.columns:
                continue

            series = df[column].dropna()
            if series.empty:
                continue
            if transform:
                series = transform(series)

            data.append(series)
            positions.append(base_positions[i] + offsets[j])
            colors.append(flow_colors[flow_id])
            flows_with_data.add(flow_id)

    if not data:
        print(f"   ⚠️  No hay datos válidos para {title}.")
        return

    width = group_width / max(len(flow_ids), 1) * 0.9
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showmeans=True,
        showfliers=show_anomalies,
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=4,
        ),
        medianprops=dict(color="black", linewidth=1.2),
    )

    whisker_colors = [c for c in colors for _ in range(2)]
    cap_colors = [c for c in colors for _ in range(2)]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.65)
    for median in box["medians"]:
        median.set_color("black")
    for whisker, color in zip(box["whiskers"], whisker_colors):
        whisker.set_color(color)
    for cap, color in zip(box["caps"], cap_colors):
        cap.set_color(color)
    for mean, color in zip(box["means"], colors):
        mean.set_markerfacecolor(color)
        mean.set_markeredgecolor("black")

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(base_positions, [sc["label"] for sc in scenarios], rotation=15)
    plt.xlabel("Escenario")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(base_positions) > 0:
        plt.xlim(base_positions[0] - group_width, base_positions[-1] + group_width)

    legend_flows = [fid for fid in flow_ids if fid in flows_with_data]
    handles = [
        plt.Line2D([0], [0], color=flow_colors[fid], lw=4) for fid in legend_flows
    ]
    labels = [merged_labels.get(fid, fid) for fid in legend_flows]
    if handles:
        plt.legend(
            handles,
            labels,
            title="Flujos",
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
        )

    out_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Guardado {out_path}")


def plot_delay_boxplots(cfg, output_dir):
    plot_cfg_raw = cfg.get("plots", {}).get("delay_boxplot", {})
    plot_cfgs = expand_plot_configs(plot_cfg_raw)

    printed_header = False
    for idx, plot_cfg in enumerate(plot_cfgs, start=1):
        if not plot_cfg.get("enabled", False):
            continue
        if not printed_header:
            print("▶ Generando boxplots de delay agrupados...")
            printed_header = True

        g_label = plot_cfg.get("label")
        title = "Delay por flujo y escenario"
        if g_label:
            title += f" - {g_label}"

        filename = plot_cfg.get("filename")
        if not filename:
            if g_label:
                filename = f"delay_boxplot_{g_label.replace(' ', '_')}.png"
            else:
                filename = f"delay_boxplot_{idx}.png"

        flows_raw = plot_cfg.get("flows") or plot_cfg.get("include")
        flow_filter, flow_labels = normalize_flow_filter(flows_raw)

        plot_grouped_boxplot(
            cfg,
            output_dir,
            column="mean",
            title=title,
            ylabel="Retardo medio [ms]",
            filename=filename,
            transform=lambda s: s * 1000,
            show_anomalies=plot_cfg.get("anomal_values", False),
            flow_filter=flow_filter,
            flow_labels_override=flow_labels,
        )


def plot_jitter_boxplots(cfg, output_dir):
    plot_cfg_raw = cfg.get("plots", {}).get("jitter_boxplot", {})
    plot_cfgs = expand_plot_configs(plot_cfg_raw)

    printed_header = False
    for idx, plot_cfg in enumerate(plot_cfgs, start=1):
        if not plot_cfg.get("enabled", False):
            continue
        if not printed_header:
            print("▶ Generando boxplots de jitter agrupados...")
            printed_header = True

        g_label = plot_cfg.get("label")
        title = "Jitter por flujo y escenario"
        if g_label:
            title += f" - {g_label}"

        filename = plot_cfg.get("filename")
        if not filename:
            if g_label:
                filename = f"jitter_boxplot_{g_label.replace(' ', '_')}.png"
            else:
                filename = f"jitter_boxplot_{idx}.png"

        flows_raw = plot_cfg.get("flows") or plot_cfg.get("include")
        flow_filter, flow_labels = normalize_flow_filter(flows_raw)

        plot_grouped_boxplot(
            cfg,
            output_dir,
            column="jitter_mean",
            title=title,
            ylabel="Jitter medio [ms]",
            filename=filename,
            transform=lambda s: s * 1000,
            show_anomalies=plot_cfg.get("anomal_values", False),
            flow_filter=flow_filter,
            flow_labels_override=flow_labels,
        )


def plot_loss_bars(cfg, output_dir):
    if not cfg.get("plots", {}).get("loss_bar", {}).get("enabled", False):
        return

    print("▶ Generando barras de pérdidas totales...")
    scenarios = cfg["scenarios"]

    all_flows = collect_all_flows(cfg)

    scen_labels = [sc["label"] for sc in scenarios]
    flow_ids = list(all_flows.keys())
    flow_colors = build_flow_colors(flow_ids)

    losses = np.zeros((len(scenarios), len(flow_ids)), dtype=float)

    for j, flow_id in enumerate(flow_ids):
        for i, sc in enumerate(scenarios):
            res_dir = sc["results_dir"]
            df = load_flow_csv(res_dir, flow_id)
            if df is None or "lost" not in df.columns:
                continue
            total_lost = df["lost"].clip(lower=0).sum()
            losses[i, j] = total_lost

    x = range(len(scenarios))
    width = 0.8 / max(1, len(flow_ids))  # barras agrupadas

    plt.figure(figsize=(8, 5))
    for j, flow_id in enumerate(flow_ids):
        offset = (j - (len(flow_ids) - 1) / 2) * width
        plt.bar(
            [xi + offset for xi in x],
            losses[:, j],
            width=width,
            label=all_flows[flow_id],
            color=flow_colors.get(flow_id),
        )

    plt.xticks(list(x), scen_labels, rotation=15)
    plt.ylabel("Pérdidas totales [paquetes]")
    plt.title("Pérdidas totales por flujo y escenario")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    out_path = output_dir / "loss_totals_bar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   ✅ Guardado {out_path}")


def plot_throughput_bar(cfg, output_dir, throughput_scale=None):
    if not cfg.get("plots", {}).get("throughput_bar", {}).get("enabled", False):
        return

    print("▶ Generando barras de throughput promedio...")
    scenarios = cfg["scenarios"]

    all_flows = collect_all_flows(cfg)
    scen_labels = [sc["label"] for sc in scenarios]
    flow_ids = list(all_flows.keys())
    flow_colors = build_flow_colors(flow_ids)

    factor, unit = throughput_scale or determine_throughput_scale(cfg)
    thr = np.zeros((len(scenarios), len(flow_ids)), dtype=float)

    for j, flow_id in enumerate(flow_ids):
        for i, sc in enumerate(scenarios):
            res_dir = sc["results_dir"]
            df = load_flow_csv(res_dir, flow_id)
            if df is None or "throughput_bytes" not in df.columns:
                continue
            thr[i, j] = df["throughput_bytes"].mean() * 8 / factor

    x = range(len(scenarios))
    width = 0.8 / max(1, len(flow_ids))

    plt.figure(figsize=(8, 5))
    for j, flow_id in enumerate(flow_ids):
        offset = (j - (len(flow_ids) - 1) / 2) * width
        plt.bar(
            [xi + offset for xi in x],
            thr[:, j],
            width=width,
            label=all_flows[flow_id],
            color=flow_colors.get(flow_id),
        )

    plt.xticks(list(x), scen_labels, rotation=15)
    plt.ylabel(f"Throughput medio [{unit}]")
    plt.title("Throughput promedio por flujo y escenario")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    out_path = output_dir / "throughput_avg_bar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   ✅ Guardado {out_path}")


def plot_throughput_timeseries(cfg, output_dir, throughput_scale=None):
    t_cfg = cfg.get("plots", {}).get("throughput_timeseries", {})
    if not t_cfg.get("enabled", False):
        return

    scen_name = t_cfg["scenario"]
    flows_cfg = t_cfg.get("flows", [])
    if not flows_cfg:
        print("   ⚠️  throughput_timeseries: no hay 'flows' definidos en plots.yml")
        return

    # buscar el escenario
    scen = None
    for sc in cfg["scenarios"]:
        if sc["name"] == scen_name:
            scen = sc
            break
    if scen is None:
        print(f"   ⚠️  Escenario {scen_name} no encontrado.")
        return

    res_dir = scen["results_dir"]
    factor, unit = throughput_scale or determine_throughput_scale(cfg)
    palette = build_flow_colors(list(collect_all_flows(cfg).keys()))

    plt.figure(figsize=(10, 6))

    alguna_serie = False
    for fcfg in flows_cfg:
        flow_id = fcfg["id"]
        flow_label = fcfg.get("label", flow_id)

        df = load_flow_csv(res_dir, flow_id)
        if df is None:
            continue
        if "t_sec" not in df.columns or "throughput_bytes" not in df.columns:
            print(f"   ⚠️  {flow_id}.csv sin columnas t_sec/throughput_bytes, se omite.")
            continue

        # Ordenar por tiempo por si acaso
        df = df.sort_values("t_sec")

        serie = df["throughput_bytes"] * 8 / factor
        plt.plot(
            df["t_sec"],
            serie,
            label=flow_label,
            color=palette.get(flow_id),
        )
        alguna_serie = True

    if not alguna_serie:
        print("   ⚠️  No se pudo dibujar ninguna serie de throughput.")
        plt.close()
        return

    plt.xlabel("Tiempo [s]")
    plt.ylabel(f"Throughput [{unit}]")
    plt.title(f"Throughput vs tiempo - escenario {scen['label']}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_path = output_dir / f"throughput_timeseries_{scen_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   ✅ Guardado {out_path}")



# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser(description="Generar gráficas a partir de CSVs de flujos.")
    parser.add_argument(
        "--config",
        "-c",
        default="plots.yml",
        help="Ruta al YAML de configuración de gráficas (por defecto: plots.yml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_output_dir(cfg.get("output_dir", "figuras"))
    throughput_scale = determine_throughput_scale(cfg)

    plot_delay_boxplots(cfg, output_dir)
    plot_jitter_boxplots(cfg, output_dir)
    plot_loss_bars(cfg, output_dir)
    plot_throughput_bar(cfg, output_dir, throughput_scale)
    plot_throughput_timeseries(cfg, output_dir, throughput_scale)


if __name__ == "__main__":
    main()
