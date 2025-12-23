#!/usr/bin/env bash
set -euo pipefail

RAW="data/raw"
EXTRACTED="data/extracted"
OUT="data/processed/containers"

python3 scripts/10_extract_zips.py --raw-root "$RAW" --out-root "$EXTRACTED"
python3 scripts/20_agg_containers.py --extracted-root "$EXTRACTED" --out-dir "$OUT" --config configs/experiments.yml
echo "DONE."
