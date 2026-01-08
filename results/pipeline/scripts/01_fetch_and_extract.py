#!/usr/bin/env python3
import argparse
import subprocess
import sys
import zipfile
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import yaml

TIME_TOLERANCE_S = 2.0  # tolerancia para comparar mtimes

def sh(cmd: list[str], check=True, capture_output=False, text=True):
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_rfc3339(ts: str) -> datetime:
    # Ej: "2025-12-18T21:39:03.123456Z"
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)

def sanitize_label(label: str) -> str:
    clean = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in label)
    clean = clean.strip("_")
    return clean or "src"

def normalize_drive_paths(sc: dict) -> list[str]:
    if "drive_paths" in sc and sc.get("drive_paths") is not None:
        paths = sc.get("drive_paths")
    else:
        paths = sc.get("drive_path")

    if paths is None:
        raise ValueError("Se requiere 'drive_paths' (lista) o 'drive_path' (string) en cada escenario.")

    if isinstance(paths, str):
        paths = [paths]

    normed = []
    for p in paths:
        if not p:
            continue
        normed.append(p.strip().strip("/"))

    if not normed:
        raise ValueError("drive_paths/drive_path no pueden estar vacíos.")

    return normed

def get_remote_files_lsjson(src: str, include_glob: str):
    # Devuelve lista de dicts con Name/Path/Size/ModTime
    cmd = [
        "rclone", "lsjson",
        src,
        "--files-only",
        "--recursive",
        "--include", include_glob
    ]
    r = sh(cmd, check=True, capture_output=True)
    try:
        return json.loads(r.stdout)
    except Exception as e:
        print("ERROR: no pude parsear JSON de rclone lsjson.")
        print(r.stdout[:1000])
        raise e

def plan_downloads(src: str, dst_raw: Path, include_glob: str):
    """
    Compara remoto vs local y devuelve:
      - to_download: lista de paths relativos (ej. "1a.zip")
      - summary: dict con conteos
    """
    remote = get_remote_files_lsjson(src, include_glob)
    to_download = []
    missing = 0
    updated = 0
    same = 0

    for f in remote:
        rel = f.get("Path") or f.get("Name")
        if not rel:
            continue

        r_size = f.get("Size", None)
        r_mtime = f.get("ModTime", None)
        r_mtime_dt = parse_rfc3339(r_mtime) if r_mtime else None

        local_path = dst_raw / rel
        if not local_path.exists():
            to_download.append(rel)
            missing += 1
            continue

        st = local_path.stat()
        l_size = st.st_size
        l_mtime_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)

        size_diff = (r_size is not None) and (int(r_size) != int(l_size))
        time_newer = False
        if r_mtime_dt is not None:
            time_newer = (r_mtime_dt.timestamp() - l_mtime_dt.timestamp()) > TIME_TOLERANCE_S

        if size_diff or time_newer:
            to_download.append(rel)
            updated += 1
        else:
            same += 1

    summary = {
        "remote_total": len(remote),
        "local_missing": missing,
        "local_outdated": updated,
        "local_up_to_date": same,
        "need_download": len(to_download),
    }
    return to_download, summary

def zip_signature(zip_path: Path) -> str:
    st = zip_path.stat()
    return f"size={st.st_size};mtime={int(st.st_mtime)}"

def max_rep_index(load_dir: Path) -> int:
    max_idx = 0
    if not load_dir.exists():
        return max_idx

    for rep_dir in load_dir.iterdir():
        if not rep_dir.is_dir():
            continue
        if rep_dir.name.startswith("."):
            continue
        if not rep_dir.name.startswith("rep_"):
            continue
        try:
            idx = int(rep_dir.name.split("_", 1)[1])
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    return max_idx

def find_rep_dirs(base_dir: Path) -> list[Path]:
    reps = [p for p in sorted(base_dir.iterdir()) if p.is_dir() and p.name.startswith("rep_")]
    if reps:
        return reps

    # Fallback: si hay un único subdir, busca rep_* adentro
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        nested = [p for p in sorted(subdirs[0].iterdir()) if p.is_dir() and p.name.startswith("rep_")]
        return nested

    return []

def merge_zip_reps(zp: Path, load_dir: Path, source_label: str, dry_run: bool = False):
    """
    Extrae rep_* de un zip y los fusiona en load_dir renumerando para evitar colisiones.
    Devuelve (added_reps, skipped_zip) donde skipped_zip=1 si el zip ya estaba procesado sin cambios.
    """
    markers_dir = load_dir / ".markers"
    sig = zip_signature(zp)
    marker_file = markers_dir / f"{source_label}__{zp.stem}.json"

    prev_marker = None
    if marker_file.exists():
        try:
            prev_marker = json.loads(marker_file.read_text())
        except Exception:
            prev_marker = None

    prev_sig = (prev_marker or {}).get("signature")
    prev_reps = (prev_marker or {}).get("assigned_reps", [])

    if prev_sig == sig:
        print(f"    = {zp.name} ({source_label}): sin cambios, se omite")
        return 0, 1

    if prev_reps:
        print(f"    - {zp.name} ({source_label}): zip cambió, eliminando reps previas ({len(prev_reps)})")
        for rep_name in prev_reps:
            rep_path = load_dir / rep_name
            if rep_path.exists():
                print(f"      borrando {rep_path}")
                if not dry_run:
                    shutil.rmtree(rep_path, ignore_errors=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        with zipfile.ZipFile(zp, "r") as z:
            z.extractall(tmp_dir)

        rep_dirs = find_rep_dirs(tmp_dir)
        if not rep_dirs:
            print(f"    WARNING: {zp.name} ({source_label}) no contiene carpetas rep_*")
            return 0, 0

        start_idx = max_rep_index(load_dir)
        assigned = []
        for rep_dir in rep_dirs:
            start_idx += 1
            new_name = f"rep_{start_idx}"
            dest = load_dir / new_name
            print(f"    + {zp.name}/{rep_dir.name} ({source_label}) -> {new_name}")
            if not dry_run:
                shutil.move(str(rep_dir), dest)
            assigned.append(new_name)

    if not dry_run:
        markers_dir.mkdir(parents=True, exist_ok=True)
        marker_data = {
            "signature": sig,
            "source_label": source_label,
            "zip_name": zp.name,
            "assigned_reps": assigned,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        marker_file.write_text(json.dumps(marker_data, indent=2))

    return len(assigned), 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yml")
    ap.add_argument("--only", default="", help="Nombre de escenario a procesar (opcional)")
    ap.add_argument("--dry-run", action="store_true", help="Solo muestra acciones, no descarga ni extrae")
    ap.add_argument("--check-only", action="store_true", help="Solo verifica qué falta/qué está viejo, no descarga ni extrae")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: no existe config: {cfg_path}")
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())

    method = cfg.get("drive", {}).get("method", "rclone")
    remote = cfg.get("drive", {}).get("remote", "").strip()
    if method != "rclone":
        print("ERROR: por ahora solo soporta method=rclone.")
        sys.exit(1)
    if not remote:
        print("ERROR: drive.remote está vacío en el YAML.")
        sys.exit(1)

    raw_root = Path(cfg.get("paths", {}).get("raw_root", "data/raw")).resolve()
    extracted_root = Path(cfg.get("paths", {}).get("extracted_root", "data/extracted")).resolve()
    safe_mkdir(raw_root)
    safe_mkdir(extracted_root)

    zip_glob = cfg.get("defaults", {}).get("zip_glob", "*.zip")
    rclone_flags = cfg.get("defaults", {}).get("rclone_flags", [])

    scenarios = cfg.get("scenarios", [])
    if args.only:
        scenarios = [s for s in scenarios if s.get("name") == args.only]
        if not scenarios:
            print(f"ERROR: escenario '{args.only}' no está en el YAML.")
            sys.exit(1)

    # Verifica rclone
    if not (args.dry_run or args.check_only):
        try:
            sh(["rclone", "version"], check=True)
        except Exception:
            print("ERROR: rclone no está instalado o no está en PATH.")
            sys.exit(1)

    for sc in scenarios:
        name = sc["name"]
        try:
            drive_paths = normalize_drive_paths(sc)
        except ValueError as e:
            print(f"ERROR en escenario '{name}': {e}")
            sys.exit(1)

        sc_zip_glob = sc.get("zip_glob", zip_glob)

        dst_raw = raw_root / name
        dst_extracted = extracted_root / name
        safe_mkdir(dst_raw)
        safe_mkdir(dst_extracted)

        print(f"\n=== Escenario: {name} ===")
        print(f"Drive paths: {drive_paths}")
        print(f"Local raw base: {dst_raw}")
        print(f"Local extracted: {dst_extracted}")
        print(f"zip_glob: {sc_zip_glob}")

        drive_entries = []
        for idx, dp in enumerate(drive_paths, start=1):
            dp_clean = dp.strip().strip("/")
            src = f"{remote}{dp_clean}"
            label_tail = sanitize_label(Path(dp_clean).name)
            source_label = f"drive{idx}_{label_tail}" if len(drive_paths) > 1 else "drive1"
            raw_target = dst_raw if len(drive_paths) == 1 else dst_raw / source_label
            safe_mkdir(raw_target)

            print(f"\n-- Drive #{idx}: {dp_clean}")
            print(f"   src: {src}")
            print(f"   raw_dir: {raw_target}")

            if args.dry_run:
                print("(dry-run) saltando verificación remota.")
                to_download = []
                summary = {}
            else:
                to_download, summary = plan_downloads(src, raw_target, sc_zip_glob)
                print("   Verificación:")
                for k, v in summary.items():
                    print(f"     - {k}: {v}")

            drive_entries.append({
                "idx": idx,
                "drive_path": dp_clean,
                "src": src,
                "raw_dir": raw_target,
                "source_label": source_label,
                "to_download": to_download,
                "summary": summary,
            })

        if args.check_only:
            continue

        # 2) Descargar SOLO si hace falta (por drive)
        for entry in drive_entries:
            if args.dry_run:
                print(f"(dry-run) no se descarga drive #{entry['idx']}.")
                continue

            to_download = entry["to_download"]
            if not to_download:
                print(f"OK: Drive #{entry['idx']} sin zips nuevos/actualizados. No se ejecuta rclone copy.")
                continue

            with tempfile.NamedTemporaryFile("w", delete=False) as tf:
                for rel in to_download:
                    tf.write(rel + "\n")
                file_list = tf.name

            cmd = ["rclone", "copy", entry["src"], str(entry["raw_dir"]),
                   "--files-from", file_list]
            cmd += rclone_flags
            sh(cmd, check=True)

        # 3) Extraer fusionando repeticiones (renumera para evitar colisiones)
        added_reps_total = 0
        skipped_zips_total = 0
        for entry in drive_entries:
            raw_dir = entry["raw_dir"]
            zips = sorted(raw_dir.glob(sc_zip_glob))
            if not zips:
                print(f"WARNING: No hay zips en {raw_dir} con glob {sc_zip_glob}")
                continue

            for zp in zips:
                load_label = zp.stem
                out_dir = dst_extracted / load_label
                safe_mkdir(out_dir)

                if args.dry_run:
                    print(f"(dry-run) fusionaría {zp.name} ({entry['source_label']}) en {out_dir}")
                    continue

                added, skipped = merge_zip_reps(zp, out_dir, entry["source_label"], dry_run=args.dry_run)
                added_reps_total += added
                skipped_zips_total += skipped

        if args.dry_run:
            print(f"OK (dry-run): {name}: revisadas {len(drive_entries)} rutas de Drive.")
        else:
            print(f"OK: {name}: reps nuevas/agregadas={added_reps_total}, zips sin cambios={skipped_zips_total}")

    print("\nDONE.")

if __name__ == "__main__":
    main()
