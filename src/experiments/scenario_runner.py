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
import concurrent.futures
from collections import deque


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(data, path):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def move_controller_csvs(csv_root: Path, report_dir: Path):
    """
    Move controller CSVs generated alongside topology_conf into this repetition's report dir.
    """
    ovs_dir = report_dir / "ovs"
    flows_dir = report_dir / "flows"
    ovs_dir.mkdir(parents=True, exist_ok=True)
    flows_dir.mkdir(parents=True, exist_ok=True)

    link_candidates = [
        csv_root / "ovs" / "link_stats.csv",
        csv_root / "link_stats.csv",
    ]
    flow_candidates = [
        csv_root / "flows" / "flow_stats.csv",
        csv_root / "flow_stats.csv",
    ]

    for src in link_candidates:
        if src.exists():
            dest = ovs_dir / "link_stats.csv"
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))
            break

    for src in flow_candidates:
        if src.exists():
            dest = flows_dir / "flow_stats.csv"
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))
            break


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


def _bytes_gib(n: int) -> float:
    return n / (1024 ** 3)

def _disk_free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(str(path)).free
    except FileNotFoundError:
        return shutil.disk_usage(str(path.parent)).free

def _dir_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total

def _try_stage_to_shm(src_dir: Path, shm_root: Path, shm_min_free_gb: float) -> Path | None:
    """Copy a directory to /dev/shm (or another tmpfs) if there is enough space.
    Returns the staged directory path, or None if staging is not possible.
    """
    try:
        shm_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    free = _disk_free_bytes(shm_root)
    needed = _dir_size_bytes(src_dir)
    # keep a safety margin
    if free < needed + int(shm_min_free_gb * (1024 ** 3)):
        return None
    staged = shm_root / f"{src_dir.parent.name}_{src_dir.name}"
    # Ensure unique (in case of reruns)
    if staged.exists():
        shutil.rmtree(staged, ignore_errors=True)
    try:
        shutil.copytree(src_dir, staged, dirs_exist_ok=False)
        return staged
    except Exception:
        try:
            shutil.rmtree(staged, ignore_errors=True)
        except Exception:
            pass
        return None

def _run_flow_resolver(resolver_cmd: list[str], resolver_log: Path) -> int:
    """Run flow_resolver and write combined stdout/stderr to resolver_log."""
    resolver_log.parent.mkdir(parents=True, exist_ok=True)
    with open(resolver_log, "w") as lf:
        proc = subprocess.Popen(resolver_cmd, stdout=lf, stderr=lf, text=True)
        return proc.wait()

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
        "--modify-scenarios-csv",
        action="store_true",
        help="Generate scaled scenarios and overwrite scenarios CSV; if not set, use scenarios as-is.",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python command to run flow_resolver.py.",
    )
    parser.add_argument(
        "--no-analyze-flows",
        action="store_true",
        help="Skip flow_resolver analysis (default: disabled).",
    )

    parser.add_argument(
        "--pipeline-analyze",
        action="store_true",
        help="Pipeline: run flow_resolver in background so next simulation can start immediately.",
    )
    parser.add_argument(
        "--max-analysis-jobs",
        type=int,
        default=1,
        help="Max concurrent flow_resolver subprocesses when --pipeline-analyze is enabled.",
    )
    parser.add_argument(
        "--max-waiting-runs",
        type=int,
        default=0,
        help=(
            "Max number of finished repetitions allowed to wait for analysis. "
            "0 is safest for disk (no backlog) but may add small waits if analysis is slower than simulation."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=60.0,
        help="Pause launching a new repetition if free disk (on output filesystem) drops below this value.",
    )
    parser.add_argument(
        "--stage-pcaps-to-shm",
        action="store_true",
        help="If possible, copy PCAPs to /dev/shm for analysis and delete disk copy early (uses RAM).",
    )
    parser.add_argument(
        "--shm-min-free-gb",
        type=float,
        default=5.0,
        help="Safety margin to keep free on /dev/shm when staging PCAPs.",
    )
    parser.add_argument(
        "--analysis-nice",
        type=int,
        default=10,
        help="nice level for background analysis (higher = less priority). Used only with --pipeline-analyze.",
    )
    parser.add_argument(
        "--analysis-ionice",
        action="store_true",
        help="Use ionice idle class for background analysis (Linux only). Used only with --pipeline-analyze.",
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
    base_qos_section = copy.deepcopy(base_topology.get("qos")) if isinstance(base_topology, dict) else None
    # ensure capture_conf exists/valid
    load_yaml(capture_conf_path)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.repetitions <= 0:
        raise ValueError("repetitions must be >= 1")

    if args.modify_scenarios_csv:
        scenario_col, container_cols, scenarios = generate_scaled_scenarios(
            args.scenarios_csv, args.alpha_scale, args.alpha_interval
        )
    else:
        # Leer escenarios tal cual del CSV sin modificarlo
        with open(args.scenarios_csv, newline="") as f:
            reader = list(csv.DictReader(f))
        if not reader:
            raise ValueError(f"{args.scenarios_csv} está vacío")
        scenario_col = reader[0].keys().__iter__().__next__()
        container_cols = [c for c in reader[0].keys() if c != scenario_col]
        scenarios = reader

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
        if base_qos_section is not None:
            topo_cfg["qos"] = copy.deepcopy(base_qos_section)
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
            logs_dir = rep_dir / "logs"
            capture_dir.mkdir(parents=True, exist_ok=True)
            flows_dir.mkdir(parents=True, exist_ok=True)
            containers_dir.mkdir(parents=True, exist_ok=True)
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
            ]

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

            move_controller_csvs(scenario_dir, report_dir)

            pcaps = list(capture_dir.glob("*.pcap"))
            if not pcaps:
                warn_msg = f"⚠️ [run {rep}] No pcap files found in {capture_dir}"
                print(warn_msg)
                (logs_dir / "warnings.log").write_text(warn_msg)
                if args.abort_on_error:
                    raise RuntimeError(warn_msg)
                continue


            # --- Flow analysis (sync by default, pipeline in background if enabled) ---
            if args.no_analyze_flows:
                print(f"ℹ️ [run {rep}/{args.repetitions}] Flow analysis skipped (--no-analyze-flows).")
            else:
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

                if not args.pipeline_analyze:
                    print(f"📊 [run {rep}/{args.repetitions}] Generating reports:", " ".join(resolver_cmd))
                    resolver_log = logs_dir / "resolver_command.log"
                    resolver_rc = _run_flow_resolver(resolver_cmd, resolver_log)
                    if resolver_rc != 0:
                        msg = (
                            f"❌ [run {rep}] flow_resolver failed (code {resolver_rc}); "
                            f"see {resolver_log}"
                        )
                        print(msg)
                        if args.abort_on_error:
                            raise RuntimeError(msg)
                        continue
                else:
                    # Lazily create executor + state on first use
                    if ("analysis_executor" not in locals()) or (analysis_executor is None):
                        analysis_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_analysis_jobs))
                        analysis_futures = deque()

                    # Backpressure: avoid unbounded backlog (disk safety)
                    max_inflight = max(1, args.max_analysis_jobs + max(0, args.max_waiting_runs))
                    # Wait if too many inflight analyses
                    while len([f for f in analysis_futures if not f.done()]) >= max_inflight:
                        done, _ = concurrent.futures.wait(list(analysis_futures), return_when=concurrent.futures.FIRST_COMPLETED)
                        # prune finished futures to keep deque small
                        analysis_futures = deque([f for f in analysis_futures if not f.done()])

                    # Disk safety: wait if low free space on output filesystem
                    out_root = Path(args.output_root)
                    while _bytes_gib(_disk_free_bytes(out_root)) < args.min_free_gb:
                        print(f"🧯 [run {rep}] Low disk space (<{args.min_free_gb} GiB free). Waiting for analysis cleanup...")
                        done, _ = concurrent.futures.wait(list(analysis_futures), return_when=concurrent.futures.FIRST_COMPLETED)
                        analysis_futures = deque([f for f in analysis_futures if not f.done()])
                        if not analysis_futures:
                            break

                    # Optional: stage PCAPs to /dev/shm (RAM) and delete disk copy early
                    staged_dir = None
                    pcaps_for_resolver = capture_dir
                    if args.stage_pcaps_to_shm:
                        shm_root = Path("/dev/shm") / "containernet_pcaps"
                        staged_dir = _try_stage_to_shm(capture_dir, shm_root, args.shm_min_free_gb)
                        if staged_dir is not None:
                            pcaps_for_resolver = staged_dir
                            try:
                                shutil.rmtree(capture_dir, ignore_errors=True)
                            except Exception:
                                pass

                    # Update resolver cmd to use staged dir if applicable
                    resolver_cmd = resolver_cmd.copy()
                    # replace --pcaps arg value
                    for i, tok in enumerate(resolver_cmd):
                        if tok == "--pcaps" and i + 1 < len(resolver_cmd):
                            resolver_cmd[i + 1] = str(pcaps_for_resolver)
                            break

                    # Run analysis in background with lower priority (nice/ionice)
                    def _analysis_task(cmd, log_path, include_pcaps, orig_capture_dir, staged_dir_path):
                        # Apply nice/ionice if requested and available
                        final_cmd = cmd
                        if args.analysis_ionice:
                            final_cmd = ["ionice", "-c3"] + final_cmd
                        if args.analysis_nice is not None:
                            final_cmd = ["nice", "-n", str(args.analysis_nice)] + final_cmd

                        rc = _run_flow_resolver(final_cmd, log_path)

                        # Cleanup PCAPs if they should not be kept in artifacts
                        if not include_pcaps:
                            if staged_dir_path is not None and Path(staged_dir_path).exists():
                                shutil.rmtree(staged_dir_path, ignore_errors=True)
                            if orig_capture_dir is not None and Path(orig_capture_dir).exists():
                                shutil.rmtree(orig_capture_dir, ignore_errors=True)
                        return rc

                    resolver_log = logs_dir / "resolver_command.log"
                    print(f"🧵 [run {rep}/{args.repetitions}] Queuing background analysis -> {resolver_log}")
                    fut = analysis_executor.submit(
                        _analysis_task,
                        resolver_cmd,
                        resolver_log,
                        args.include_pcaps,
                        str(capture_dir),
                        str(staged_dir) if staged_dir is not None else None,
                    )
                    analysis_futures.append(fut)
            # Opcional: limpiar PCAPs de la repetición si no se solicitaron.
            # En modo pipeline, el cleanup ocurre al finalizar el análisis en background.
            if (not args.pipeline_analyze) and (not args.include_pcaps) and capture_dir.exists():
                subprocess.run(["rm", "-rf", str(capture_dir)], check=False)


        # If pipeline mode is enabled, wait for background analyses to finish before zipping/uploading
        if args.pipeline_analyze and "analysis_futures" in locals():
            pending = [f for f in analysis_futures if not f.done()]
            if pending:
                print(f"🧾 [scenario {label}] Waiting for {len(pending)} background analysis job(s) to finish...")
                concurrent.futures.wait(pending)
            # After all done, shut down executor to release resources
            try:
                analysis_executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
            analysis_executor = None
            analysis_futures = deque()
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
