#!/usr/bin/env bash

############################################
#   DOCKER CONTAINER METRICS CAPTURE
#
#   Usage:
#   ./capture_metrics_containers.sh c1 c2 ... -t 60 -o dir -i 1
#
#   -t / --time     Duration in seconds (default 60)
#   -o / --output   Output directory (default ./metrics)
#   -i / --interval Sampling interval in seconds (default 1)
############################################

# --- Default values ---
DOCKER=(sudo docker)
DURATION=60
OUTPUT_DIR="./metrics"
INTERVAL=1
CONTAINERS=()

# --- Function: convert Docker size string (e.g. "12.3MiB") to bytes ---
to_bytes() {
    local val="$1"

    # Empty or "0" case
    if [ -z "$val" ] || [ "$val" = "0" ]; then
        echo 0
        return
    fi

    # Separate numeric part and unit part
    local num unit factor
    num=$(echo "$val" | sed 's/[^0-9\.]//g')
    unit=$(echo "$val" | sed 's/[0-9\.]//g')

    # Normalize some common unit variants
    case "$unit" in
        B|"")
            factor=1
            ;;
        kB|KB)
            factor=1000
            ;;
        KiB)
            factor=1024
            ;;
        MB)
            factor=1000000
            ;;
        MiB)
            factor=1048576
            ;;
        GB)
            factor=1000000000
            ;;
        GiB)
            factor=1073741824
            ;;
        TB)
            factor=1000000000000
            ;;
        TiB)
            factor=1099511627776
            ;;
        *)
            # Unknown unit: fallback to 1
            factor=1
            ;;
    esac

    # Use bc to handle decimals
    # Output as integer
    echo "($num * $factor)/1" | bc
}

# --- Argument parsing ---
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
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 <cont1> <cont2> ... -t <seconds> -o <dir> [-i interval]"
            exit 1
            ;;
        *)
            CONTAINERS+=("$1")
            shift
            ;;
    esac
done

# --- Validation ---
if [ ${#CONTAINERS[@]} -eq 0 ]; then
    echo "Usage: $0 <cont1> <cont2> ... -t <seconds> -o <dir> [-i interval]"
    echo "Example: $0 ryu_ctrl containernet h1 h2 -t 300 -o ./metrics -i 1"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- Wait until all containers are running ---
wait_for_containers() {
    local pending=("${CONTAINERS[@]}")
    local attempts=0
    local max_attempts=100

    echo "Waiting for containers to be running..."

    while [ ${#pending[@]} -gt 0 ]; do
        local next_pending=()
        for c in "${pending[@]}"; do
            # Check that the container exists and is running
            if "${DOCKER[@]}" inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q true; then
                echo "✅ Container is running: $c"
            else
                next_pending+=("$c")
            fi
        done

        if [ ${#next_pending[@]} -eq 0 ]; then
            break
        fi

        attempts=$((attempts + 1))
        if [ "$attempts" -ge "$max_attempts" ]; then
            echo "Not all containers were running after ${max_attempts} attempts."
            echo "Pending containers: ${next_pending[*]}"
            exit 1
        fi

        pending=("${next_pending[@]}")
        sleep 2
    done
}

wait_for_containers

echo "============================================="
echo "  STARTING CONTAINER METRICS CAPTURE"
echo "  Containers: ${CONTAINERS[*]}"
echo "  Duration:   ${DURATION}s"
echo "  Interval:   ${INTERVAL}s"
echo "  Folder:     $OUTPUT_DIR"
echo "============================================="

# --- Create per-container CSV files with header ---
for c in "${CONTAINERS[@]}"; do
    CSV_FILE="${OUTPUT_DIR}/${c}.csv"
    echo "timestamp,cpu_perc,mem_used_bytes,mem_limit_bytes,mem_perc,net_rx_bytes,net_tx_bytes,block_read_bytes,block_write_bytes,pids" > "$CSV_FILE"
done

# --- Sampling loop ---
end_time=$(( $(date +%s) + DURATION ))

while [ "$(date +%s)" -lt "$end_time" ]; do
    TS=$(date +%s)

    for c in "${CONTAINERS[@]}"; do
        CSV_FILE="${OUTPUT_DIR}/${c}.csv"

        # Get stats only for this container
        STATS=$("${DOCKER[@]}" stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}" "$c" 2>/dev/null)

        # If the container is gone or stats are not available, skip this sample
        if [ -z "$STATS" ]; then
            continue
        fi

        IFS=',' read -r cpu memusage memperc netio blockio pids <<< "$STATS"

        # cpu: e.g. "3.45%" -> remove '%'
        cpu_clean=$(echo "$cpu" | tr -d '%')
        mem_perc_clean=$(echo "$memperc" | tr -d '%')

        # memusage: "50MiB / 1GiB"
        mem_used_raw=$(echo "$memusage" | awk '{print $1}')
        mem_limit_raw=$(echo "$memusage" | awk '{print $3}')
        mem_used_bytes=$(to_bytes "$mem_used_raw")
        mem_limit_bytes=$(to_bytes "$mem_limit_raw")

        # netio: "12.3kB / 4.5MB"  => RX / TX
        net_rx_raw=$(echo "$netio" | awk '{print $1}')
        net_tx_raw=$(echo "$netio" | awk '{print $3}')
        net_rx_bytes=$(to_bytes "$net_rx_raw")
        net_tx_bytes=$(to_bytes "$net_tx_raw")

        # blockio: "0B / 0B"       => READ / WRITE
        block_read_raw=$(echo "$blockio" | awk '{print $1}')
        block_write_raw=$(echo "$blockio" | awk '{print $3}')
        block_read_bytes=$(to_bytes "$block_read_raw")
        block_write_bytes=$(to_bytes "$block_write_raw")

        echo "$TS,$cpu_clean,$mem_used_bytes,$mem_limit_bytes,$mem_perc_clean,$net_rx_bytes,$net_tx_bytes,$block_read_bytes,$block_write_bytes,$pids" >> "$CSV_FILE"
    done

    sleep "$INTERVAL"
done

echo "Metrics stored in: $OUTPUT_DIR"
echo "============================================="
