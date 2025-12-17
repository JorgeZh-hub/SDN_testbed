#!/bin/bash

############################################
# v2: CAPTURA DE MÚLTIPLES INTERFACES
#     sin bloquear por la interfaz más lenta.
#
# Cambios clave:
#  - Ya NO espera a que estén *todas* las interfaces para empezar.
#  - Lanza un "worker" por interfaz que espera y luego hace `exec tcpdump`.
#    (el PID que guardas siempre corresponde al tcpdump final).
#
# Uso:
#   ./capture_links_v2.sh if1 if2 ... -o dir -s 96
############################################

OUTPUT_DIR="./capturas"
SNAPLEN=0
INTERFACES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) OUTPUT_DIR="$2"; shift 2;;
    -s|--snaplen) SNAPLEN="$2"; shift 2;;
    -*) echo "Opción desconocida: $1"; exit 1;;
    *) INTERFACES+=("$1"); shift;;
  esac
done

if [ ${#INTERFACES[@]} -eq 0 ]; then
  echo "Uso: $0 <iface1> <iface2> ... -o <dir> [-s snaplen]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo "  INICIANDO CAPTURAS (v2)"
echo "  Interfaces: ${INTERFACES[*]}"
echo "  Carpeta:    $OUTPUT_DIR"
if [ "$SNAPLEN" -gt 0 ]; then
  echo "  Truncado:   ${SNAPLEN} bytes por paquete"
else
  echo "  Truncado:   NO (captura completa)"
fi
echo "============================================="

PIDS=()

cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap 'cleanup' EXIT
trap 'cleanup; exit 0' INT TERM

start_one() {
  local IFACE="$1"
  local PCAP_FILE="${OUTPUT_DIR}/${IFACE}.pcap"
  echo "➡️  [${IFACE}] esperando interfaz y capturando → $PCAP_FILE"

  # El PID de este subshell será el PID del tcpdump, porque hacemos exec.
  (
    while [ ! -d "/sys/class/net/$IFACE" ]; do
      sleep 1
    done
    echo "✅ [${IFACE}] interfaz disponible, arrancando tcpdump"
    if [ "$SNAPLEN" -gt 0 ]; then
      exec tcpdump -i "$IFACE" -s "$SNAPLEN" -w "$PCAP_FILE" >/dev/null 2>&1
    else
      exec tcpdump -i "$IFACE" -w "$PCAP_FILE" >/dev/null 2>&1
    fi
  ) &
  PIDS+=($!)
}

for IFACE in "${INTERFACES[@]}"; do
  start_one "$IFACE"
done

wait