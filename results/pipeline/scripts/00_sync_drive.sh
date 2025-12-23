#!/usr/bin/env bash
set -euo pipefail

# Uso: ./scripts/00_sync_drive.sh gdrive:MiCarpetaResultados data/raw

SRC="${1:-}"
DST="${2:-}"

if [[ -z "$SRC" || -z "$DST" ]]; then
  echo "Uso: $0 <rclone_src> <local_dst>"
  echo "Ej:  $0 gdrive:ResultadosSDN data/raw"
  exit 1
fi

mkdir -p "$DST"
rclone sync "$SRC" "$DST" --progress --checksum
