#!/usr/bin/env bash
set -euo pipefail


############################################
# v2: DOCKER CONTAINER METRICS CAPTURE
#
# Problema de v1:
#   Espera a que *todos* los contenedores estén running y recién ahí empieza
#   a muestrear. Si 1 contenedor tarda 70s en aparecer, pierdes 70s de métricas
#   de los demás.
#
# Solución v2:
#   - No bloquea por el último contenedor.
#   - Muestrea cada contenedor en cuanto esté disponible.
#   - Opcional: --wait-all para comportamiento v1.
#
# Uso:
#   ./capture_containers_metrics_v2.sh c1 c2 ... -o dir -i 1 [--wait-all]
############################################

## ---- DOCKER helper ----
# Preferimos NO usar sudo si el usuario ya tiene acceso a docker (grupo docker)
# o si el script se ejecuta como root. Si se requiere sudo, usamos -n
# (no interactivo) para evitar bloqueos en background.
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  DOCKER=(docker)
else
  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
  else
    DOCKER=(sudo -n docker)
    if ! sudo -n true >/dev/null 2>&1; then
      echo "ERROR: docker requiere sudo, pero no hay sesión sudo válida (sudo -n falla)." >&2
      echo "Solución rápida: ejecuta run_experiment desde una terminal y acepta la contraseña una sola vez (sudo -v)," >&2
      echo "o añade tu usuario al grupo docker (recomendado), o configura NOPASSWD para docker." >&2
      exit 1
    fi
  fi
fi
OUTPUT_DIR="./metrics"
INTERVAL=1
WAIT_ALL=0
CONTAINERS=()

to_bytes() {
  local val="$1"
  if [ -z "$val" ] || [ "$val" = "0" ]; then
    echo 0
    return
  fi
  local num unit factor
  num=$(echo "$val" | sed 's/[^0-9\.]//g')
  unit=$(echo "$val" | sed 's/[0-9\.]//g')
  case "$unit" in
    B|"") factor=1;;
    kB|KB) factor=1000;;
    KiB) factor=1024;;
    MB) factor=1000000;;
    MiB) factor=1048576;;
    GB) factor=1000000000;;
    GiB) factor=1073741824;;
    TB) factor=1000000000000;;
    TiB) factor=1099511627776;;
    *) factor=1;;
  esac
  echo "($num * $factor)/1" | bc
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) OUTPUT_DIR="$2"; shift 2;;
    -i|--interval) INTERVAL="$2"; shift 2;;
    --wait-all) WAIT_ALL=1; shift;;
    -*)
      echo "Unknown option: $1"
      echo "Usage: $0 <cont1> <cont2> ... -o <dir> [-i interval] [--wait-all]"
      exit 1
      ;;
    *) CONTAINERS+=("$1"); shift;;
  esac
done

if [ ${#CONTAINERS[@]} -eq 0 ]; then
  echo "Usage: $0 <cont1> <cont2> ... -o <dir> [-i interval] [--wait-all]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ "$WAIT_ALL" -eq 1 ]; then
  echo "[v2] Waiting for ALL containers to be running (legacy mode)..."
  pending=("${CONTAINERS[@]}")
  attempts=0
  max_attempts=100
  while [ ${#pending[@]} -gt 0 ]; do
    next_pending=()
    for c in "${pending[@]}"; do
      if "${DOCKER[@]}" inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q true; then
        echo "✅ Container is running: $c"
      else
        next_pending+=("$c")
      fi
    done
    [ ${#next_pending[@]} -eq 0 ] && break
    attempts=$((attempts+1))
    if [ "$attempts" -ge "$max_attempts" ]; then
      echo "Not all containers were running after ${max_attempts} attempts. Pending: ${next_pending[*]}"
      exit 1
    fi
    pending=("${next_pending[@]}")
    sleep 2
  done
else
  echo "[v2] Non-blocking mode: each container starts being sampled as soon as it exists."
fi

echo "============================================="
echo "  STARTING CONTAINER METRICS CAPTURE (v2)"
echo "  Containers: ${CONTAINERS[*]}"
echo "  Interval:   ${INTERVAL}s"
echo "  Folder:     $OUTPUT_DIR"
echo "============================================="

for c in "${CONTAINERS[@]}"; do
  CSV_FILE="${OUTPUT_DIR}/${c}.csv"
  echo "timestamp,cpu_perc,mem_used_bytes,mem_limit_bytes,mem_perc,net_rx_bytes,net_tx_bytes,block_read_bytes,block_write_bytes,pids" > "$CSV_FILE"
done

cleanup_children() {
  local pids
  pids=$(jobs -p)
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
    wait $pids 2>/dev/null || true
  fi
}
trap 'cleanup_children' EXIT
trap 'cleanup_children; exit 0' INT TERM

while true; do
  TS=$(date +%s)

  for c in "${CONTAINERS[@]}"; do
    CSV_FILE="${OUTPUT_DIR}/${c}.csv"

    STATS=$("${DOCKER[@]}" stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}" "$c" 2>/dev/null || true)
    [ -z "$STATS" ] && continue

    IFS=',' read -r cpu memusage memperc netio blockio pids <<< "$STATS"
    cpu_clean=$(echo "$cpu" | tr -d '%')
    mem_perc_clean=$(echo "$memperc" | tr -d '%')

    mem_used_raw=$(echo "$memusage" | awk '{print $1}')
    mem_limit_raw=$(echo "$memusage" | awk '{print $3}')
    mem_used_bytes=$(to_bytes "$mem_used_raw")
    mem_limit_bytes=$(to_bytes "$mem_limit_raw")

    net_rx_raw=$(echo "$netio" | awk '{print $1}')
    net_tx_raw=$(echo "$netio" | awk '{print $3}')
    net_rx_bytes=$(to_bytes "$net_rx_raw")
    net_tx_bytes=$(to_bytes "$net_tx_raw")

    block_read_raw=$(echo "$blockio" | awk '{print $1}')
    block_write_raw=$(echo "$blockio" | awk '{print $3}')
    block_read_bytes=$(to_bytes "$block_read_raw")
    block_write_bytes=$(to_bytes "$block_write_raw")

    echo "$TS,$cpu_clean,$mem_used_bytes,$mem_limit_bytes,$mem_perc_clean,$net_rx_bytes,$net_tx_bytes,$block_read_bytes,$block_write_bytes,$pids" >> "$CSV_FILE"
  done

  sleep "$INTERVAL"
done