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
DEFAULT_CAPTURE_CONF_REL="src/experiments/capture_conf.yml"

#############################################
# Docker images and container names        #
#############################################

DOCKER=(sudo docker)
KILL=(kill)
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
PARSE_CONTROLLER_SCRIPT="$SCRIPT_DIR/parse_controller_from_top.py"
PARSE_CAPTURE_SCRIPT="$SCRIPT_DIR/parse_capture_conf.py"

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
CAPTURE_CONF_REL="${CAPTURE_CONF:-$DEFAULT_CAPTURE_CONF_REL}"
# Absolute path to topology on host (used to read controller cfg)
if [[ "$TOPOLOGY_FILE_REL" = /* ]]; then
    TOPOLOGY_FILE_HOST="$TOPOLOGY_FILE_REL"
else
    TOPOLOGY_FILE_HOST="$TESTBED_DIR/$TOPOLOGY_FILE_REL"
fi
# Path passed to Containernet container (remap to /sim when under project root)
if [[ "$TOPOLOGY_FILE_HOST" == "$TESTBED_DIR"* ]]; then
    TOPOLOGY_FILE_IN_CONTAINER="/sim${TOPOLOGY_FILE_HOST#$TESTBED_DIR}"
else
    TOPOLOGY_FILE_IN_CONTAINER="$TOPOLOGY_FILE_REL"
fi
# Absolute path to capture conf on host (used to derive capture targets)
if [[ "$CAPTURE_CONF_REL" = /* ]]; then
    CAPTURE_CONF_HOST="$CAPTURE_CONF_REL"
else
    CAPTURE_CONF_HOST="$TESTBED_DIR/$CAPTURE_CONF_REL"
fi

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --duration, -d <sec>    Total duration of the experiment (default: ${TOTAL_DURATION}s).
  --capture-dir <dir>     Directory for PCAP output (default: ${CAPTURE_DIR}).
  --snaplen <bytes>       Snaplen for tcpdump (default: full capture).
  --log-dir <dir>         Directory for logs (default: ${LOG_DIR}).
  --startup-buffer <s>    (unused) reserved buffer time (default: ${STARTUP_BUFFER}s).
  --metrics-dir <dir>     Directory for container metrics CSV (default: ${METRICS_DIR}).
  --metrics-interval <s>  Sampling interval for metrics (default: ${METRICS_INTERVAL}s).
  --links-metrics-dir <d> Directory for OVS link metrics CSV (default: ${LINKS_METRICS_DIR}).
  --links-interval <s>    Sampling interval for link metrics (default: ${LINKS_INTERVAL}s).
  --queue-analyzer        Enable queue analyzer in OVS metrics.
  --capture-conf <file>   Capture configuration YAML (default: ${CAPTURE_CONF_REL}).
  --help                  Show this help.
EOF
}

#############################################
# Argument parsing                         #
#############################################

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --metrics-interval)
            METRICS_INTERVAL="$2"
            shift 2
            ;;
        --links-metrics-dir)
            LINKS_METRICS_DIR="$2"
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
        --capture-conf)
            CAPTURE_CONF_REL="$2"
            # recompute absolute host path after parsing all args
            shift 2
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

# Recompute absolute capture conf path after parsing args
if [[ "$CAPTURE_CONF_REL" = /* ]]; then
    CAPTURE_CONF_HOST="$CAPTURE_CONF_REL"
else
    CAPTURE_CONF_HOST="$TESTBED_DIR/$CAPTURE_CONF_REL"
fi

#############################################
# Derive capture targets from capture_conf  #
#############################################

unique_append() {
    # $1 array name, $2 value
    local -n _arr="$1"
    local val="$2"
    for x in "${_arr[@]}"; do
        if [[ "$x" == "$val" ]]; then
            return
        fi
    done
    _arr+=("$val")
}

if [[ -f "$CAPTURE_CONF_HOST" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        case "$line" in
            IFACE=*)
                val="${line#IFACE=}"
                unique_append INTERFACES "$val"
                ;;
            CONTAINER=*)
                val="${line#CONTAINER=}"
                unique_append METRICS_CONTAINERS "$val"
                ;;
            LINK=*)
                val="${line#LINK=}"
                unique_append LINKS "$val"
                ;;
        esac
    done < <(
        HOST_PROJECT_ROOT="$TESTBED_DIR" python3 "$PARSE_CAPTURE_SCRIPT" \
            --capture-conf "$CAPTURE_CONF_HOST" \
            --topology "$TOPOLOGY_FILE_HOST"
    )
fi

if [[ ${#INTERFACES[@]} -eq 0 ]]; then
    echo "Error: no interfaces to capture. Provide --iface or ensure capture_conf is valid."
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

stop_process() {
    local pid="$1"
    local label="$2"
    local timeout="${3:-10}"

    if [[ -z "${pid}" ]]; then
        return
    fi
    if ! ps -p "${pid}" >/dev/null 2>&1; then
        return
    fi

    echo "[${label}] Stopping..."
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

    local waited=0
    while ps -p "${pid}" >/dev/null 2>&1; do
        if [[ "${waited}" -ge "${timeout}" ]]; then
            echo "[${label}] Forcing stop..."
            kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    wait "${pid}" 2>/dev/null || true
}

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
    stop_process "${CAPTURE_PID}" "CAPTURE"
    stop_process "${METRICS_PID}" "METRICS"
    stop_process "${OVS_METRICS_PID}" "OVS"

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
    local ctrl_name="$RYU_CONTAINER_NAME"
    local ctrl_image="$RYU_IMAGE"
    local ctrl_cmd="--observe-links /app/app.py"
    local ctrl_volumes=("${CONTROLLER_DIR}/baseline:/app")
    local custom_volumes=0

    if [[ -f "$TOPOLOGY_FILE_HOST" ]]; then
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            local key="${line%%=*}"
            local val="${line#*=}"
            case "$key" in
                NAME)
                    ctrl_name="$val"
                    ;;
                IMAGE)
                    ctrl_image="$val"
                    ;;
                COMMAND)
                    ctrl_cmd="$val"
                    ;;
                VOLUME)
                    if [[ $custom_volumes -eq 0 ]]; then
                        ctrl_volumes=()
                    fi
                    custom_volumes=1
                    ctrl_volumes+=("$val")
                    ;;
            esac
        done < <(
            HOST_PROJECT_ROOT="$TESTBED_DIR" python3 "$PARSE_CONTROLLER_SCRIPT" "$TOPOLOGY_FILE_HOST"
        )
    else
        echo "[CTRL] Warning: topology file not found at ${TOPOLOGY_FILE_HOST}; using defaults." | tee -a "$log_file"
    fi

    # update globals so cleanup uses the resolved controller name
    RYU_CONTAINER_NAME="$ctrl_name"
    RYU_IMAGE="$ctrl_image"

    (
        echo "[CTRL] Working directory (host): $CONTROLLER_DIR"
        echo "[CTRL] Image: $RYU_IMAGE"
        echo "[CTRL] Container name: $RYU_CONTAINER_NAME"
        echo "[CTRL] Command: ${ctrl_cmd:-'(image entrypoint)'}"
        if [[ ${#ctrl_volumes[@]} -gt 0 ]]; then
            echo "[CTRL] Volumes:"
            for v in "${ctrl_volumes[@]}"; do
                echo "       - $v"
            done
        fi
    ) | tee -a "$log_file"

    local run_cmd=("${DOCKER[@]}" run -d --rm --name "$RYU_CONTAINER_NAME" -w /app --net=host -e PYTHONPATH=/app)
    for v in "${ctrl_volumes[@]}"; do
        run_cmd+=(-v "$v")
    done
    run_cmd+=("$RYU_IMAGE")

    if [[ -n "$ctrl_cmd" ]]; then
        # shellcheck disable=SC2206
        local ctrl_cmd_arr=($ctrl_cmd)
        run_cmd+=("${ctrl_cmd_arr[@]}")
    fi

    "${run_cmd[@]}" >>"$log_file" 2>&1

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
    cmd+=(-o "$CAPTURE_DIR")
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
    cmd+=(-o "$METRICS_DIR" -i "$METRICS_INTERVAL")

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
    cmd+=(-o "$LINKS_METRICS_DIR" -i "$LINKS_INTERVAL")
    if [[ "$QUEUE_ANALYZER" -eq 1 ]]; then
        cmd+=(--queue-analyzer)
    fi

    local log_file="${LOG_DIR}/ovs_metrics_${RUN_ID}.log"
    "${cmd[@]}" >"$log_file" 2>&1 &
    OVS_METRICS_PID=$!

    echo "[OVS] Capture is running (PID $OVS_METRICS_PID). Log: ${log_file}"
}

stop_captures() {
    stop_process "${CAPTURE_PID}" "CAPTURE"
    CAPTURE_PID=""
}

stop_metrics() {
    stop_process "${METRICS_PID}" "METRICS"
    METRICS_PID=""
}

stop_ovs_metrics() {
    stop_process "${OVS_METRICS_PID}" "OVS"
    OVS_METRICS_PID=""
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
        echo "[SIM] TOPOLOGY_FILE (env): $TOPOLOGY_FILE_IN_CONTAINER"
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
        -e TOPOLOGY_FILE="${TOPOLOGY_FILE_IN_CONTAINER}" \
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
echo "Startup buffer:        ${STARTUP_BUFFER}s (unused)"
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
start_captures
sleep 1
start_metrics
sleep 1
start_ovs_metrics

echo "[MAIN] Running experiment for ${TOTAL_DURATION}s..."
sleep "$TOTAL_DURATION"

# Stop captures before tearing down topology
stop_captures
stop_metrics
stop_ovs_metrics
echo "[MAIN] Captures and metrics stopped."

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