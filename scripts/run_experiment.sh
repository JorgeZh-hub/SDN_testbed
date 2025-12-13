#!/bin/bash
set -euo pipefail

#############################################
# Auto-detected paths (no hardcoded $HOME) #
#############################################

# Directory where this script lives (assumed: .../Red/testbed/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Testbed root (one level above scripts/, i.e. .../Red/testbed)
TESTBED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Controller project directory: .../Red/testbed/containers/core/controller
CONTROLLER_DIR="$TESTBED_DIR/containers/core/controller"

# Topology script and config inside testbed
TOPOLOGY_SCRIPT_REL="src/topology/topology_test.py"
DEFAULT_TOPOLOGY_FILE_REL="src/topology/topology_conf.yml"

#############################################
# Docker images and container names        #
#############################################

DOCKER=(sudo docker)
KILL=(sudo kill)
RYU_IMAGE="controller_ryu"
RYU_CONTAINER_NAME="ryu-ctrl"

CONTAINERNET_IMAGE="containernet_docker"
CONTAINERNET_CONTAINER_NAME="containernet-sim"

#############################################
# Other scripts (host-side)                #
#############################################

CAPTURE_SCRIPT="$TESTBED_DIR/scripts/capture_links.sh"
METRICS_SCRIPT="$TESTBED_DIR/scripts/capture_containers_metrics.sh"
OVS_METRICS_SCRIPT="$TESTBED_DIR/scripts/capture_ovs_metrics.sh"
STOP_CONTAINERS="$TESTBED_DIR/scripts/stop_containers.sh"

#############################################
# Default parameters                       #
#############################################

TOTAL_DURATION=120
STARTUP_BUFFER=100
CAPTURE_DIR="$TESTBED_DIR/results/captures"
CAPTURE_SNAPLEN=0
INTERFACES=()
LOG_DIR="$TESTBED_DIR/results/logs"
METRICS_DIR="$TESTBED_DIR/results/metrics"
METRICS_CONTAINERS=()
METRICS_INTERVAL=1
LINKS_METRICS_DIR="$TESTBED_DIR/results/link_metrics"
LINKS=()
LINKS_INTERVAL=1
QUEUE_ANALYZER=0
TOPOLOGY_FILE_REL="${TOPOLOGY_FILE:-$DEFAULT_TOPOLOGY_FILE_REL}"

usage() {
    cat <<EOF
Usage: $0 --iface <if1> [--iface <if2> ...] [options]

Options:
  --iface, -i <iface>     Interface to capture (can be repeated).
  --duration, -d <sec>    Total duration of the experiment (default: ${TOTAL_DURATION}s).
  --capture-dir <dir>     Directory for PCAP output (default: ${CAPTURE_DIR}).
  --snaplen <bytes>       Snaplen for tcpdump (default: full capture).
  --log-dir <dir>         Directory for logs (default: ${LOG_DIR}).
  --metrics-dir <dir>     Directory for container metrics CSV (default: ${METRICS_DIR}).
  --container, -c <name>  Container to capture metrics (can be repeated).
  --metrics-interval <s>  Sampling interval for metrics (default: ${METRICS_INTERVAL}s).
  --links-metrics-dir <d> Directory for OVS link metrics CSV (default: ${LINKS_METRICS_DIR}).
  --link, -l "<sw1-if1, sw2-if2, cap_mbps>" Link to monitor (can be repeated). Capacity is used for utilization%.
  --links-interval <s>    Sampling interval for link metrics (default: ${LINKS_INTERVAL}s).
  --queue-analyzer        Enable queue analyzer in OVS metrics.
  --help                  Show this help.
EOF
}

#############################################
# Argument parsing                         #
#############################################

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--iface)
            INTERFACES+=("$2")
            shift 2
            ;;
        -d|--duration)
            TOTAL_DURATION="$2"
            shift 2
            ;;
        --capture-dir)
            CAPTURE_DIR="$2"
            shift 2
            ;;
        --snaplen)
            CAPTURE_SNAPLEN="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --metrics-dir)
            METRICS_DIR="$2"
            shift 2
            ;;
        -c|--container)
            METRICS_CONTAINERS+=("$2")
            shift 2
            ;;
        --metrics-interval)
            METRICS_INTERVAL="$2"
            shift 2
            ;;
        --links-metrics-dir)
            LINKS_METRICS_DIR="$2"
            shift 2
            ;;
        -l|--link)
            LINKS+=("$2")
            shift 2
            ;;
        --links-interval)
            LINKS_INTERVAL="$2"
            shift 2
            ;;
        --queue-analyzer)
            QUEUE_ANALYZER=1
            shift
            ;;
        --startup-buffer)
            STARTUP_BUFFER="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ${#INTERFACES[@]} -eq 0 ]]; then
    echo "Error: you must specify at least one interface with --iface."
    usage
    exit 1
fi

#############################################
# Prepare directories and run ID           #
#############################################

mkdir -p "$LOG_DIR" "$CAPTURE_DIR" "$METRICS_DIR" "$LINKS_METRICS_DIR"
RUN_ID=$(date +"%Y%m%d_%H%M%S")

CAPTURE_PID=""
METRICS_PID=""
OVS_METRICS_PID=""
SIM_STARTED=0
STOPPED_CONTAINERS=0
CTRL_LOG_PID=""
SIM_LOG_PID=""

#############################################
# Cleanup handler                          #
#############################################

cleanup() {
    echo "[CLEANUP] Starting cleanup sequence..."

    # Stop log tails
    if [[ -n "${CTRL_LOG_PID}" ]] && "${KILL[@]}" -0 "$CTRL_LOG_PID" 2>/dev/null; then
        "${KILL[@]}" "$CTRL_LOG_PID" 2>/dev/null || true
        wait "$CTRL_LOG_PID" 2>/dev/null || true
    fi
    if [[ -n "${SIM_LOG_PID}" ]] && "${KILL[@]}" -0 "$SIM_LOG_PID" 2>/dev/null; then
        "${KILL[@]}" "$SIM_LOG_PID" 2>/dev/null || true
        wait "$SIM_LOG_PID" 2>/dev/null || true
    fi

    # Stop captures if still running
    if [[ -n "${CAPTURE_PID}" ]] && "${KILL[@]}" -0 "$CAPTURE_PID" 2>/dev/null; then
        echo "[CLEANUP] Stopping packet capture..."
        "${KILL[@]}" "$CAPTURE_PID" 2>/dev/null || true
        wait "$CAPTURE_PID" 2>/dev/null || true
    fi
    if [[ -n "${METRICS_PID}" ]] && "${KILL[@]}" -0 "$METRICS_PID" 2>/dev/null; then
        echo "[CLEANUP] Stopping metrics capture..."
        "${KILL[@]}" "$METRICS_PID" 2>/dev/null || true
        wait "$METRICS_PID" 2>/dev/null || true
    fi
    if [[ -n "${OVS_METRICS_PID}" ]] && "${KILL[@]}" -0 "$OVS_METRICS_PID" 2>/dev/null; then
        echo "[CLEANUP] Stopping OVS link metrics capture..."
        "${KILL[@]}" "$OVS_METRICS_PID" 2>/dev/null || true
        wait "$OVS_METRICS_PID" 2>/dev/null || true
    fi

    # Stop Containernet container if running
    if "${DOCKER[@]}" ps -q -f "name=^${CONTAINERNET_CONTAINER_NAME}$" >/dev/null; then
        echo "[CLEANUP] Stopping Containernet container ${CONTAINERNET_CONTAINER_NAME}..."
        "${DOCKER[@]}" stop "${CONTAINERNET_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    # Optional: run additional cleanup script (host-side)
    if [[ $SIM_STARTED -eq 1 && $STOPPED_CONTAINERS -eq 0 && -x "$STOP_CONTAINERS" ]]; then
        echo "[CLEANUP] Running stop_containers.sh..."
        "$STOP_CONTAINERS" || true
        STOPPED_CONTAINERS=1
    fi

    # Stop RYU controller container if running
    if "${DOCKER[@]}" ps -q -f "name=^${RYU_CONTAINER_NAME}$" >/dev/null; then
        echo "[CLEANUP] Stopping RYU controller container ${RYU_CONTAINER_NAME}..."
        "${DOCKER[@]}" stop "${RYU_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    echo "[CLEANUP] Cleanup sequence finished."
}
trap cleanup EXIT INT TERM

#############################################
# Start RYU controller (in container)      #
#############################################

start_controller() {
    echo "[CTRL] Starting RYU controller container..."
    local log_file="${LOG_DIR}/controller_${RUN_ID}.log"

    (
        echo "[CTRL] Working directory (host): $CONTROLLER_DIR"
        echo "[CTRL] Image: $RYU_IMAGE"
        echo "[CTRL] Container name: $RYU_CONTAINER_NAME"
        echo "[CTRL] Command: ryu-manager --observe-links /app/multipath.py"
    ) | tee -a "$log_file"

    "${DOCKER[@]}" run -d --rm \
        --name "${RYU_CONTAINER_NAME}" \
        --net=host \
        -v "${CONTROLLER_DIR}":/app \
        "${RYU_IMAGE}" \
        --observe-links /app/multipath.py >>"$log_file" 2>&1

    if ! "${DOCKER[@]}" ps -q -f "name=^${RYU_CONTAINER_NAME}$" >/dev/null; then
        echo "[CTRL] Error: failed to start RYU controller container. See ${log_file}."
        exit 1
    fi

    echo "[CTRL] RYU controller container is running (logs: ${log_file})"

    # Stream logs in background to the same file
    "${DOCKER[@]}" logs -f "${RYU_CONTAINER_NAME}" >>"$log_file" 2>&1 &
    CTRL_LOG_PID=$!
}

#############################################
# Start tcpdump captures                   #
#############################################

start_captures() {
    echo "[CAPTURE] Starting captures on interfaces: ${INTERFACES[*]}"
    local cmd=(sudo "$CAPTURE_SCRIPT")
    cmd+=("${INTERFACES[@]}")
    cmd+=(-t "$TOTAL_DURATION" -o "$CAPTURE_DIR")
    if [[ "$CAPTURE_SNAPLEN" -gt 0 ]]; then
        cmd+=(-s "$CAPTURE_SNAPLEN")
    fi

    local log_file="${LOG_DIR}/capture_${RUN_ID}.log"
    local attempt=1

    while true; do
        echo "[CAPTURE] Attempt #${attempt}" | tee -a "$log_file"
        "${cmd[@]}" >"$log_file" 2>&1 &
        CAPTURE_PID=$!

        sleep 2

        if ! "${KILL[@]}" -0 "$CAPTURE_PID" 2>/dev/null; then
            status=$?
            echo "[CAPTURE] Capture process exited immediately (code $status)." | tee -a "$log_file"
            if [[ $attempt -ge 5 ]]; then
                echo "[CAPTURE] Maximum number of retries reached. Aborting."
                exit 1
            fi
            attempt=$((attempt + 1))
            sleep 3
            continue
        fi

        echo "[CAPTURE] Capture is running (PID $CAPTURE_PID). Log: ${log_file}" | tee -a "$log_file"
        break
    done
}

#############################################
# Start container metrics capture          #
#############################################

start_metrics() {
    if [[ ${#METRICS_CONTAINERS[@]} -eq 0 ]]; then
        return
    fi
    echo "[METRICS] Starting capture for containers: ${METRICS_CONTAINERS[*]}"
    local cmd=(sudo "$METRICS_SCRIPT")
    cmd+=("${METRICS_CONTAINERS[@]}")
    cmd+=(-t "$TOTAL_DURATION" -o "$METRICS_DIR" -i "$METRICS_INTERVAL")

    local log_file="${LOG_DIR}/metrics_${RUN_ID}.log"
    "${cmd[@]}" >"$log_file" 2>&1 &
    METRICS_PID=$!

    echo "[METRICS] Capture is running (PID $METRICS_PID). Log: ${log_file}"
}

#############################################
# Start OVS link metrics capture           #
#############################################

start_ovs_metrics() {
    if [[ ${#LINKS[@]} -eq 0 ]]; then
        return
    fi

    echo "[OVS] Starting link metrics capture for links: ${LINKS[*]}"
    local cmd=(sudo "$OVS_METRICS_SCRIPT")
    cmd+=("${LINKS[@]}")
    cmd+=(-t "$TOTAL_DURATION" -o "$LINKS_METRICS_DIR" -i "$LINKS_INTERVAL")
    if [[ "$QUEUE_ANALYZER" -eq 1 ]]; then
        cmd+=(--queue-analyzer)
    fi

    local log_file="${LOG_DIR}/ovs_metrics_${RUN_ID}.log"
    "${cmd[@]}" >"$log_file" 2>&1 &
    OVS_METRICS_PID=$!

    echo "[OVS] Capture is running (PID $OVS_METRICS_PID). Log: ${log_file}"
}

#############################################
# Start Containernet simulation container  #
#############################################

start_simulation() {
    echo "[SIM] Starting Containernet simulation container..."
    SIM_STARTED=1
    local log_file="${LOG_DIR}/topology_${RUN_ID}.log"

    (
        echo "[SIM] Host testbed directory: $TESTBED_DIR"
        echo "[SIM] Image: $CONTAINERNET_IMAGE"
        echo "[SIM] Container name: $CONTAINERNET_CONTAINER_NAME"
        echo "[SIM] Working dir inside container: /sim"
        echo "[SIM] Topology script (relative): $TOPOLOGY_SCRIPT_REL"
        echo "[SIM] TOPOLOGY_FILE (env): $TOPOLOGY_FILE_REL"
    ) | tee -a "$log_file"


    "${DOCKER[@]}" run -d --rm \
        --name "${CONTAINERNET_CONTAINER_NAME}" \
        --privileged \
        --net=host \
        --pid=host \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /var/run/openvswitch:/var/run/openvswitch \
        -v "${TESTBED_DIR}":/sim \
        -w /sim \
        -e HOST_PROJECT_ROOT="${TESTBED_DIR}" \
        -e NON_INTERACTIVE=1 \
        -e TOPOLOGY_FILE="${TOPOLOGY_FILE_REL}" \
        -e SHELL=/bin/bash \
        --entrypoint python3 \
        "${CONTAINERNET_IMAGE}" \
        "${TOPOLOGY_SCRIPT_REL}" >>"$log_file" 2>&1

    if ! "${DOCKER[@]}" ps -q -f "name=^${CONTAINERNET_CONTAINER_NAME}$" >/dev/null; then
        echo "[SIM] Error: failed to start Containernet container. See ${log_file}."
        exit 1
    fi

    echo "[SIM] Containernet simulation container is running (logs: ${log_file})"

    # Stream logs in background to the same file
    "${DOCKER[@]}" logs -f "${CONTAINERNET_CONTAINER_NAME}" >>"$log_file" 2>&1 &
    SIM_LOG_PID=$!
}

#############################################
# Main flow                                #
#############################################

echo "========== AUTOMATED RUN =========="
echo "Interfaces to capture: ${INTERFACES[*]}"
echo "Total duration:        ${TOTAL_DURATION}s"
echo "Startup buffer:        ${STARTUP_BUFFER}s (wait before starting captures)"
echo "PCAP directory:        ${CAPTURE_DIR}"
echo "Log directory:         ${LOG_DIR}"
echo "Topology file (rel):   ${TOPOLOGY_FILE_REL}"
if [[ ${#METRICS_CONTAINERS[@]} -gt 0 ]]; then
    echo "Metrics containers:    ${METRICS_CONTAINERS[*]}"
    echo "Metrics dir:           ${METRICS_DIR} (interval ${METRICS_INTERVAL}s)"
fi
if [[ ${#LINKS[@]} -gt 0 ]]; then
    echo "OVS links:             ${LINKS[*]}"
    echo "OVS metrics dir:       ${LINKS_METRICS_DIR} (interval ${LINKS_INTERVAL}s, queue analyzer ${QUEUE_ANALYZER})"
fi
echo "==================================="

# Go to testbed directory to keep relative paths consistent
cd "$TESTBED_DIR"

start_controller
sleep 1
start_simulation
echo "[MAIN] Waiting ${STARTUP_BUFFER}s for topology to settle before starting captures..."
sleep "$STARTUP_BUFFER"
start_captures
sleep 1
start_metrics
sleep 1
start_ovs_metrics

echo "[MAIN] Waiting for captures to finish (${TOTAL_DURATION}s)..."
wait "$CAPTURE_PID" || true
CAPTURE_PID=""
if [[ -n "${METRICS_PID}" ]]; then
    wait "$METRICS_PID" || true
    METRICS_PID=""
fi
if [[ -n "${OVS_METRICS_PID}" ]]; then
    wait "$OVS_METRICS_PID" || true
    OVS_METRICS_PID=""
fi
echo "[MAIN] Captures completed."

# Stop simulation container after captures
    if "${DOCKER[@]}" ps -q -f "name=^${CONTAINERNET_CONTAINER_NAME}$" >/dev/null; then
        echo "[MAIN] Stopping Containernet simulation container..."
        "${DOCKER[@]}" stop "${CONTAINERNET_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    # Optional extra cleanup script
    if [[ -x "$STOP_CONTAINERS" ]]; then
        echo "[MAIN] Running stop_containers.sh..."
    "$STOP_CONTAINERS" || true
    STOPPED_CONTAINERS=1
fi

# Stop controller container
    if "${DOCKER[@]}" ps -q -f "name=^${RYU_CONTAINER_NAME}$" >/dev/null; then
        echo "[MAIN] Stopping RYU controller container..."
        "${DOCKER[@]}" stop "${RYU_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

echo "[MAIN] Run completed. Check logs in ${LOG_DIR}/"
