#!/usr/bin/env python3
import argparse
import csv
import copy
import subprocess
from pathlib import Path
import yaml
import os
import zipfile
import time
import shutil


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(data, path):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def slugify(text):
    safe = "".join(c if c.isalnum() or c in (".", "-", "_") else "_" for c in text.strip())
    return safe or "scenario"


def resolve_container_flow(col_name, hosts_cfg):
    """
    Given a column <container>_<flow>, return (container, flow).
    Supports flow names with underscores by checking container prefixes.
    """
    for host in sorted(hosts_cfg, key=len, reverse=True):  # longest prefixes first
        prefix = f"{host}_"
        if col_name.startswith(prefix):
            flow_name = col_name[len(prefix) :]
            if not flow_name:
                raise ValueError(f"Column {col_name} missing flow suffix after container.")
            return host, flow_name
    raise ValueError(f"Column {col_name} does not match any container defined in topology_conf.")


def generate_scaled_scenarios(csv_path, alpha_scale, alpha_interval):
    """
    Read scenario CSV, treat first row as base, and generate scaled scenarios
    (factor 1+alpha_interval ... alpha_scale) multiplying each value (rounded).
    Overwrites CSV with base+generated and returns (scenario_col, container_cols, scenarios_list).
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Scenario CSV must have headers.")
        scenario_col = reader.fieldnames[0]
        container_cols = reader.fieldnames[1:]
        rows = list(reader)

    if not rows:
        raise ValueError("Scenario CSV must contain at least one base row (first row).")

    base_row = rows[0]

    def to_int(value, col):
        value = (value or "").strip()
        if value == "":
            return 0
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid value in column {col} of base scenario: {value}") from exc

    base_values = {col: to_int(base_row.get(col, ""), col) for col in container_cols}

    scenarios = []
    # Escenario base tal cual (pero normalizado a enteros)
    normalized_base = {scenario_col: base_row.get(scenario_col, "base")}
    normalized_base.update({col: str(base_values[col]) for col in container_cols})
    scenarios.append(normalized_base)

    if alpha_scale < 1:
        raise ValueError("alpha_scale must be >= 1.0")
    if alpha_interval <= 0:
        raise ValueError("alpha_interval must be > 0")

    factor = 1.0 + alpha_interval
    # Generar escenarios hasta llegar a alpha_scale * alpha (la base es alpha)
    while factor <= alpha_scale + 1e-9:
        factor_rounded = round(factor, 4)  # evita ruido flotante
        scenario_name = f"{normalized_base[scenario_col]}_x{factor_rounded:.2f}"
        new_row = {scenario_col: scenario_name}
        for col in container_cols:
            scaled = int(round(base_values[col] * factor_rounded))
            new_row[col] = str(scaled)
        scenarios.append(new_row)
        factor += alpha_interval

    # Sobrescribir el CSV con base + generados
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[scenario_col] + container_cols)
        writer.writeheader()
        writer.writerows(scenarios)

    return scenario_col, container_cols, scenarios


def zip_repetition(
    rep_dir: Path,
    scenario_dir: Path,
    scenario_topology: Path,
    include_pcaps: bool,
    zip_name: str,
    include_topology: bool,
) -> Path:
    """
    Create a zip for a repetition. Structure inside the zip:
      - rep_X/...(reports, logs, pcaps if included)
      - topology_conf.yml only on the first repetition (include_topology=True)
    """
    zip_path = scenario_dir / zip_name
    base_root = scenario_dir  # archive paths start at rep_X/...

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add topology at root only when requested (first repetition)
        if include_topology and scenario_topology.exists():
            arc_topo = scenario_topology.relative_to(base_root)
            zf.write(scenario_topology, arcname=arc_topo)

        for root, _, files in os.walk(rep_dir):
            for fname in files:
                fpath = Path(root) / fname
                if not include_pcaps and fpath.suffix == ".pcap":
                    continue
                arcname = fpath.relative_to(base_root)
                zf.write(fpath, arcname=arcname)

    return zip_path


def zip_scenario(scenario_dir: Path, include_pcaps: bool, zip_name: str) -> Path:
    """
    Create a zip for the whole scenario directory, optionally skipping pcaps.
    """
    zip_path = scenario_dir.with_name(zip_name)
    base_root = scenario_dir

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(scenario_dir):
            for fname in files:
                fpath = Path(root) / fname
                if not include_pcaps and fpath.suffix == ".pcap":
                    continue
                arcname = fpath.relative_to(base_root)
                zf.write(fpath, arcname=arcname)

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Run load scenarios from a CSV and orchestrate captures + analysis."
    )
    parser.add_argument("--scenarios-csv", default="escenarios.csv", help="Scenario CSV file.")
    parser.add_argument(
        "--base-topology",
        default="topology_conf.yml",
        help="Base topology file to apply scenarios on.",
    )
    parser.add_argument(
        "--capture-conf",
        default="capture_conf.yml",
        help="YAML with flows/links/containers capture configuration.",
    )
    parser.add_argument(
        "--queue-analyzer",
        action="store_true",
        help="Enable queue analyzer in capture_ovs_metrics (default: disabled).",
    )
    parser.add_argument(
        "--output-root",
        default="scenario_runs",
        help="Directory to store per-scenario results.",
    )
    parser.add_argument(
        "--run-script",
        default="./scripts/run_experiment.sh",
        help="Script that launches the simulation (run_experiment.sh).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds for each scenario (passed to run_experiment).",
    )
    parser.add_argument(
        "--alpha-scale",
        type=float,
        default=1.0,
        help="Maximum scale factor over base scenario (>=1).",
    )
    parser.add_argument(
        "--alpha-interval",
        type=float,
        default=0.1,
        help="Scale step to generate scenarios from the base.",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python command to run flow_resolver.py.",
    )
    parser.add_argument(
        "-n",
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions to execute per scenario.",
    )
    parser.add_argument(
        "--abort-on-error",
        action="store_true",
        help="Stop the run if a repetition fails (default: continue and log the error).",
    )
    parser.add_argument(
        "--rclone-dest",
        default="",
        help="Optional rclone destination (e.g., remote:bucket/path) to upload a zip per repetition.",
    )
    parser.add_argument(
        "--include-pcaps",
        action="store_true",
        help="Include PCAP files in the uploaded zip (default: exclude pcaps).",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep local output (pcaps, reports, logs, zips). Default: delete after upload.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_topology_path = Path(args.base_topology)
    if not base_topology_path.is_absolute():
        base_topology_path = (script_dir / base_topology_path).resolve()

    capture_conf_path = Path(args.capture_conf)
    if not capture_conf_path.is_absolute():
        capture_conf_path = (script_dir / capture_conf_path).resolve()

    base_topology = load_yaml(base_topology_path)
    # ensure capture_conf exists/valid
    load_yaml(capture_conf_path)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.repetitions <= 0:
        raise ValueError("repetitions must be >= 1")

    scenario_col, container_cols, scenarios = generate_scaled_scenarios(
        args.scenarios_csv, args.alpha_scale, args.alpha_interval
    )

    for idx, row in enumerate(scenarios, start=1):
        scenario_name = row.get(scenario_col, f"scenario_{idx}")
        base_name = scenario_name
        factor_val = 1.0
        if "_x" in scenario_name:
            base_name, factor_part = scenario_name.split("_x", 1)
            try:
                factor_val = float(factor_part)
            except ValueError:
                factor_val = 1.0
        factor_str = f"{factor_val:g}"
        label = slugify(f"{factor_str}{base_name}")
        scenario_dir = output_root / label
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_topology = scenario_dir / "topology_conf.yml"

        topo_cfg = copy.deepcopy(base_topology)
        replacements = []
        for col in container_cols:
            value = (row.get(col, "") or "").strip()
            if value == "":
                continue
            try:
                n_proc = int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid value for {col} in scenario {scenario_name}: {value}") from exc

            container, flow_name = resolve_container_flow(col, topo_cfg["hosts"])
            host_cfg = topo_cfg["hosts"][container]
            flows_cfg = host_cfg.get("flows", [])
            matched = False
            flow_name_str = str(flow_name)
            for flow in flows_cfg:
                if str(flow.get("name")) == flow_name_str:
                    flow["n_proc"] = n_proc
                    replacements.append((container, flow_name_str, n_proc))
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Flow {flow_name} does not exist in container {container}.")

        if not replacements:
            raise RuntimeError(f"No flow replacements applied for scenario {scenario_name}; check CSV and topology.")

        write_yaml(topo_cfg, scenario_topology)

        print(f"\n🚀 Scenario: {scenario_name} (label: {label})")
        print(f"🗺️  Topology file: {scenario_topology}")
        print("🔢 n_proc replacements:", ", ".join(f"{c}.{f}={n}" for c, f, n in replacements))

        env_base = os.environ.copy()
        env_base["TOPOLOGY_FILE"] = str(scenario_topology)

        for rep in range(1, args.repetitions + 1):
            rep_dir = scenario_dir / f"rep_{rep}"
            capture_dir = rep_dir / "pcaps"
            report_dir = rep_dir / "reports"
            flows_dir = report_dir / "flows"
            containers_dir = report_dir / "containers"
            ovs_dir = report_dir / "ovs"
            logs_dir = rep_dir / "logs"
            capture_dir.mkdir(parents=True, exist_ok=True)
            flows_dir.mkdir(parents=True, exist_ok=True)
            containers_dir.mkdir(parents=True, exist_ok=True)
            ovs_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            env = env_base.copy()

            run_cmd = [
                args.run_script,
                "--capture-dir",
                str(capture_dir),
                "--duration",
                str(args.duration),
                "--log-dir",
                str(logs_dir),
                "--capture-conf",
                str(capture_conf_path),
                "--metrics-dir",
                str(containers_dir),
                "--links-metrics-dir",
                str(ovs_dir),
            ]
            if args.queue_analyzer:
                run_cmd.append("--queue-analyzer")

            print(f"🏃 [run {rep}/{args.repetitions}] Executing:", " ".join(run_cmd))
            run_log = logs_dir / "run_command.log"
            start_ts = time.time()
            with open(run_log, "w") as lf:
                proc = subprocess.Popen(run_cmd, env=env, stdout=lf, stderr=lf)
                while proc.poll() is None:
                    elapsed = int(time.time() - start_ts)
                    print(f"\r⏱️ [run {rep}/{args.repetitions}] elapsed {elapsed}s (target ~{args.duration}s)", end="", flush=True)
                    time.sleep(1)
            print()  # newline after timer
            if proc.returncode != 0:
                msg = f"❌ [run {rep}] run_experiment failed (code {proc.returncode}); see {run_log}"
                print(msg)
                if args.abort_on_error:
                    raise RuntimeError(msg)
                continue

            pcaps = list(capture_dir.glob("*.pcap"))
            if not pcaps:
                warn_msg = f"⚠️ [run {rep}] No pcap files found in {capture_dir}"
                print(warn_msg)
                (logs_dir / "warnings.log").write_text(warn_msg)
                if args.abort_on_error:
                    raise RuntimeError(warn_msg)
                continue

            resolver_cmd = [
                args.python,
                "src/analyzer/flow_resolver.py",
                "--flows",
                str(capture_conf_path),
                "--topology",
                str(scenario_topology),
                "--pcaps",
                str(capture_dir),
                "--output-dir",
                str(flows_dir),
            ]
            print(f"📊 [run {rep}/{args.repetitions}] Generating reports:", " ".join(resolver_cmd))
            resolver_log = logs_dir / "resolver_command.log"
            resolver_rc = 0
            with open(resolver_log, "w") as lf:
                proc = subprocess.Popen(resolver_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    print(line, end="")
                    lf.write(line)
                resolver_rc = proc.wait()
            if resolver_rc != 0:
                msg = (
                    f"❌ [run {rep}] flow_resolver failed (code {resolver_rc}); "
                    f"see {resolver_log}"
                )
                print(msg)
                if args.abort_on_error:
                    raise RuntimeError(msg)
                continue

            # Opcional: limpiar PCAPs de la repetición si no se solicitaron
            if not args.include_pcaps and capture_dir.exists():
                subprocess.run(["rm", "-rf", str(capture_dir)], check=False)

        # Zip y subida al final del escenario completo
        if args.rclone_dest:
            try:
                zip_name = f"{label}.zip"
                zip_path = zip_scenario(scenario_dir, include_pcaps=args.include_pcaps, zip_name=zip_name)
                dest = f"{args.rclone_dest.rstrip('/')}/{zip_path.name}"
                print(f"☁️  [scenario {label}] Uploading {zip_path} to {dest} via rclone copyto")
                upload = subprocess.run(
                    ["rclone", "copyto", str(zip_path), dest],
                    capture_output=True,
                    text=True,
                )
                (scenario_dir / "upload.log").write_text(upload.stdout + "\n" + upload.stderr)
                if upload.returncode != 0:
                    msg = f"❌ [scenario {label}] rclone copyto failed for {zip_path} -> {dest}"
                    print(msg)
                    if args.abort_on_error:
                        raise RuntimeError(msg)
                else:
                    if not args.keep_files:
                        # remove scenario artifacts after successful upload (zip and directory)
                        if scenario_dir.exists():
                            shutil.rmtree(scenario_dir, ignore_errors=True)
                        if zip_path.exists():
                            zip_path.unlink()
            except Exception as exc:
                msg = f"❌ [scenario {label}] Exception during upload: {exc}"
                print(msg)
                if args.abort_on_error:
                    raise
        elif not args.keep_files:
            # No upload: cleanup scenario directory if requested
            shutil.rmtree(scenario_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
