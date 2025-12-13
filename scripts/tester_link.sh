#!/bin/bash

# Uso:
#   ./tester_link.sh <interfaz1> [interfaz2 ...] [intervalo_segundos]
# Ejemplo:
#   ./tester_link.sh switch14-eth1 switch3-eth4 0.5

if [ "$#" -lt 1 ]; then
    echo "Uso: $0 <interfaz1> [interfaz2 ...] [intervalo_segundos]"
    echo "Ejemplo: $0 switch14-eth1 switch3-eth4 0.5"
    exit 1
fi

INTERVAL="${@: -1}"
if [[ "$INTERVAL" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    IFACES=("${@:1:$#-1}")
else
    IFACES=("$@")
    INTERVAL=1
fi

if [ "${#IFACES[@]}" -eq 0 ]; then
    echo "Debes especificar al menos una interfaz."
    exit 1
fi

for iface in "${IFACES[@]}"; do
    if [ ! -d "/sys/class/net/$iface" ]; then
        echo "Error: La interfaz '$iface' no existe."
        echo "Interfaces disponibles:"
        ls /sys/class/net
        exit 1
    fi
done

echo "Monitoreando interfaces: ${IFACES[*]} (intervalo = ${INTERVAL}s)"
echo "Presiona Ctrl+C para salir."
echo ""

declare -A RX_BEFORE TX_BEFORE

while true; do
    for iface in "${IFACES[@]}"; do
        RX_BEFORE["$iface"]=$(cat "/sys/class/net/$iface/statistics/rx_bytes")
        TX_BEFORE["$iface"]=$(cat "/sys/class/net/$iface/statistics/tx_bytes")
    done

    sleep "$INTERVAL"

    printf "\033[H\033[J"
    printf "%-15s | %15s | %15s | %15s | %15s\n" "Interfaz" "RX (Mbps)" "RX (Kbps)" "TX (Mbps)" "TX (Kbps)"
    printf "%s\n" "-------------------------------------------------------------------------------------------"

    for iface in "${IFACES[@]}"; do
        RX_AFTER=$(cat "/sys/class/net/$iface/statistics/rx_bytes")
        TX_AFTER=$(cat "/sys/class/net/$iface/statistics/tx_bytes")

        RX_DELTA=$((RX_AFTER - RX_BEFORE["$iface"]))
        TX_DELTA=$((TX_AFTER - TX_BEFORE["$iface"]))

        RX_Mbps=$(awk -v d="$RX_DELTA" -v t="$INTERVAL" 'BEGIN { printf "%.3f", (d*8)/(1000000*t) }')
        TX_Mbps=$(awk -v d="$TX_DELTA" -v t="$INTERVAL" 'BEGIN { printf "%.3f", (d*8)/(1000000*t) }')

        RX_Kbps=$(awk -v d="$RX_DELTA" -v t="$INTERVAL" 'BEGIN { printf "%.1f", (d*8)/(1000*t) }')
        TX_Kbps=$(awk -v d="$TX_DELTA" -v t="$INTERVAL" 'BEGIN { printf "%.1f", (d*8)/(1000*t) }')

        printf "%-15s | %15s | %15s | %15s | %15s\n" \
            "$iface" "$RX_Mbps" "$RX_Kbps" "$TX_Mbps" "$TX_Kbps"
    done
done
