#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path
import re

def load_label_from_zip(zip_path: Path) -> str:
    # "1a.zip" -> "1a", "1.2a.zip" -> "1.2a"
    return zip_path.stem

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_zip(zip_path: Path, out_dir: Path):
    safe_mkdir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="Carpeta con controladores y zips (data/raw)")
    ap.add_argument("--out-root", required=True, help="Carpeta destino (data/extracted)")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    safe_mkdir(out_root)

    # Esperado: raw_root/<controller>/*.zip
    for controller_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        controller_name = controller_dir.name
        zips = sorted(controller_dir.glob("*.zip"))
        if not zips:
            continue

        for zp in zips:
            load_label = load_label_from_zip(zp)
            out_dir = out_root / controller_name / load_label
            # Si ya existe, no re-extraer (puedes cambiarlo si quieres)
            marker = out_dir / ".extracted_ok"
            if marker.exists():
                continue

            safe_mkdir(out_dir)
            extract_zip(zp, out_dir)
            marker.write_text("ok\n")

    print("OK: extracción completa.")

if __name__ == "__main__":
    main()
