#!/bin/bash

############################################
#      CAPTURA DE MÚLTIPLES INTERFACES
#   con soporte para truncado (-s bytes)
#
#   Uso:
#   ./capture_links.sh if1 if2 ... -t 60 -o dir -s 96
############################################

# --- Valores por defecto ---
DURATION=60
OUTPUT_DIR="./capturas"
SNAPLEN=0          # 0 = captura completa
INTERFACES=()

# --- Parseo de argumentos ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--time)
            DURATION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--snaplen)
            SNAPLEN="$2"
            shift 2
            ;;
        -*)
            echo "Opción desconocida: $1"
            exit 1
            ;;
        *)
            INTERFACES+=("$1")
            shift
            ;;
    esac
done

# --- Validación ---
if [ ${#INTERFACES[@]} -eq 0 ]; then
    echo "Uso: $0 <iface1> <iface2> ... -t <segundos> -o <dir> [-s snaplen]"
    echo "Ejemplo: $0 switch3-eth1 switch3-eth2 -t 300 -o ./pcaps -s 128"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

wait_for_interfaces() {
    local pending=("${INTERFACES[@]}")
    local attempts=0
    local max_attempts=100

    while [ ${#pending[@]} -gt 0 ]; do
        local next_pending=()
        for iface in "${pending[@]}"; do
            if [ -d "/sys/class/net/$iface" ]; then
                echo "✅ Interfaz disponible: $iface"
            else
                next_pending+=("$iface")
            fi
        done

        if [ ${#next_pending[@]} -eq 0 ]; then
            break
        fi

        attempts=$((attempts + 1))
        if [ "$attempts" -ge "$max_attempts" ]; then
            echo "❌ No se pudieron encontrar todas las interfaces tras ${max_attempts} intentos."
            exit 1
        fi

        pending=("${next_pending[@]}")
        sleep 2
    done
}

wait_for_interfaces

echo "============================================="
echo "  INICIANDO CAPTURAS"
echo "  Interfaces: ${INTERFACES[*]}"
echo "  Duración:   ${DURATION}s"
echo "  Carpeta:    $OUTPUT_DIR"
if [ "$SNAPLEN" -gt 0 ]; then
    echo "  Truncado:   ${SNAPLEN} bytes por paquete"
else
    echo "  Truncado:   NO (captura completa)"
fi
echo "============================================="

# --- Iniciar capturas ---
PIDS=()

for IFACE in "${INTERFACES[@]}"; do
    PCAP_FILE="${OUTPUT_DIR}/${IFACE}.pcap"

    echo "➡️  Capturando $IFACE → $PCAP_FILE"

    if [ "$SNAPLEN" -gt 0 ]; then
        tcpdump -i "$IFACE" -s "$SNAPLEN" -w "$PCAP_FILE" >/dev/null 2>&1 &
    else
        tcpdump -i "$IFACE" -w "$PCAP_FILE" >/dev/null 2>&1 &
    fi

    PIDS+=($!)
done

sleep "$DURATION"

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "❌ No se pudo iniciar ninguna captura. Abortando."
    exit 1
fi

echo "🛑 Tiempo terminado. Deteniendo capturas..."
for PID in "${PIDS[@]}"; do
    kill "$PID" 2>/dev/null
done

echo "✔ Capturas guardadas en: $OUTPUT_DIR"
echo "============================================="
