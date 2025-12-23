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

def extract_zip(zip_path: Path, out_dir: Path):
    """
    Extrae limpio.
    Usa marker con firma del zip: size + mtime.
    """
    safe_mkdir(out_dir.parent)

    zip_stat = zip_path.stat()
    sig = f"size={zip_stat.st_size};mtime={int(zip_stat.st_mtime)}"
    marker = out_dir / ".extracted_ok"

    if marker.exists():
        prev = marker.read_text().strip()
        if prev == sig:
            return False  # ya extraído y zip no cambió
        # zip cambió => re-extraer limpio
        shutil.rmtree(out_dir, ignore_errors=True)

    safe_mkdir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    marker.write_text(sig + "\n")
    return True

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
        drive_path = sc["drive_path"].strip().strip("/")
        sc_zip_glob = sc.get("zip_glob", zip_glob)

        src = f"{remote}{drive_path}"
        dst_raw = raw_root / name
        dst_extracted = extracted_root / name
        safe_mkdir(dst_raw)
        safe_mkdir(dst_extracted)

        print(f"\n=== Escenario: {name} ===")
        print(f"Drive src: {src}")
        print(f"Local raw: {dst_raw}")
        print(f"Local extracted: {dst_extracted}")
        print(f"zip_glob: {sc_zip_glob}")

        # 1) Plan de descarga (verificación)
        if args.dry_run:
            print("(dry-run) saltando verificación remota.")
            to_download = []
            summary = {}
        else:
            to_download, summary = plan_downloads(src, dst_raw, sc_zip_glob)
            print("Verificación:")
            for k, v in summary.items():
                print(f"  - {k}: {v}")

        if args.check_only:
            continue

        # 2) Descargar SOLO si hace falta
        if args.dry_run:
            print("(dry-run) no se descarga.")
        else:
            if not to_download:
                print("OK: No hay zips nuevos/actualizados. No se ejecuta rclone copy.")
            else:
                # rclone copy con --files-from para copiar solo los que cambian
                with tempfile.NamedTemporaryFile("w", delete=False) as tf:
                    for rel in to_download:
                        tf.write(rel + "\n")
                    file_list = tf.name

                cmd = ["rclone", "copy", src, str(dst_raw),
                       "--files-from", file_list]
                cmd += rclone_flags
                sh(cmd, check=True)

        # 3) Extraer (re-extrae solo si el zip cambió)
        zips = sorted(dst_raw.glob(sc_zip_glob))
        if not zips:
            print(f"WARNING: No hay zips en {dst_raw} con glob {sc_zip_glob}")
            continue

        extracted_count = 0
        skipped_count = 0
        for zp in zips:
            load_label = zp.stem
            out_dir = dst_extracted / load_label

            if args.dry_run:
                print(f"(dry-run) extraería {zp.name} -> {out_dir}")
                continue

            changed = extract_zip(zp, out_dir)
            if changed:
                extracted_count += 1
            else:
                skipped_count += 1

        print(f"OK: {name}: extraídos/re-extraídos={extracted_count}, sin_cambios={skipped_count}")

    print("\nDONE.")

if __name__ == "__main__":
    main()
