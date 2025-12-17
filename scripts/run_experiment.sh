#!/bin/bash
set -euo pipefail

#############################################
# v2: Run captures UNTIL the topology ends
#     (instead of "sleep --duration")
#
# Key fixes:
#  - No fixed sleep window that truncates capture time.
#  - Waits for the Containernet container to exit via `docker wait`.
#  - Uses `timeout` as a safety net (max wall time = duration + startup-buffer + extra).
#
# NOTE: this script is a drop-in replacement for run_experiment.sh.
#############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTROLLER_DIR="$TESTBED_DIR/containers/core/controller"

TOPOLOGY_SCRIPT_REL="src/topology/topology_test.py"
DEFAULT_TOPOLOGY_FILE_REL="src/topology/topology_conf.yml"
DEFAULT_CAPTURE_CONF_REL="src/experiments/capture_conf.yml"

# Se decide dinámicamente más abajo (para evitar sudo innecesario)
DOCKER=(docker)
KILL=(kill)
RYU_IMAGE="controller_ryu"
RYU_CONTAINER_NAME="ryu-ctrl"
CONTAINERNET_IMAGE="containernet_docker"
CONTAINERNET_CONTAINER_NAME="containernet-sim"

CAPTURE_SCRIPT="$TESTBED_DIR/scripts/capture_links.sh"
METRICS_SCRIPT="$TESTBED_DIR/scripts/capture_containers_metrics.sh"
OVS_METRICS_SCRIPT="$TESTBED_DIR/scripts/capture_ovs_metrics.sh"
STOP_CONTAINERS="$TESTBED_DIR/scripts/stop_containers.sh"
PARSE_CONTROLLER_SCRIPT="$SCRIPT_DIR/parse_controller_from_top.py"
PARSE_CAPTURE_SCRIPT="$SCRIPT_DIR/parse_capture_conf.py"

TOTAL_DURATION=120
STARTUP_BUFFER=100
MAX_EXTRA_WALL=120   # extra seconds beyond duration+startup (parsing/teardown/jitter)

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

if [[ "$TOPOLOGY_FILE_REL" = /* ]]; then
  TOPOLOGY_FILE_HOST="$TOPOLOGY_FILE_REL"
else
  TOPOLOGY_FILE_HOST="$TESTBED_DIR/$TOPOLOGY_FILE_REL"
fi

if [[ "$TOPOLOGY_FILE_HOST" == "$TESTBED_DIR"* ]]; then
  TOPOLOGY_FILE_IN_CONTAINER="/sim${TOPOLOGY_FILE_HOST#$TESTBED_DIR}"
else
  TOPOLOGY_FILE_IN_CONTAINER="$TOPOLOGY_FILE_REL"
fi

if [[ "$CAPTURE_CONF_REL" = /* ]]; then
  CAPTURE_CONF_HOST="$CAPTURE_CONF_REL"
else
  CAPTURE_CONF_HOST="$TESTBED_DIR/$CAPTURE_CONF_REL"
fi

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --duration, -d <sec>    Target experiment duration (traffic window) (default: ${TOTAL_DURATION}s).
  --startup-buffer <s>    Expected max setup delay before traffic (default: ${STARTUP_BUFFER}s).
  --max-extra-wall <s>    Extra wall-time safety margin (default: ${MAX_EXTRA_WALL}s).
  --capture-dir <dir>     Directory for PCAP output (default: ${CAPTURE_DIR}).
  --snaplen <bytes>       Snaplen for tcpdump (default: full capture).
  --log-dir <dir>         Directory for logs (default: ${LOG_DIR}).
  --metrics-dir <dir>     Directory for container metrics CSV (default: ${METRICS_DIR}).
  --metrics-interval <s>  Sampling interval for metrics (default: ${METRICS_INTERVAL}s).
  --links-metrics-dir <d> Directory for OVS link metrics CSV (default: ${LINKS_METRICS_DIR}).
  --links-interval <s>    Sampling interval for link metrics (default: ${LINKS_INTERVAL}s).
  --queue-analyzer        Enable queue analyzer in OVS metrics.
  --capture-conf <file>   Capture configuration YAML (default: ${CAPTURE_CONF_REL}).
  --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--duration) TOTAL_DURATION="$2"; shift 2;;
    --startup-buffer) STARTUP_BUFFER="$2"; shift 2;;
    --max-extra-wall) MAX_EXTRA_WALL="$2"; shift 2;;
    --capture-dir) CAPTURE_DIR="$2"; shift 2;;
    --snaplen) CAPTURE_SNAPLEN="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --metrics-dir) METRICS_DIR="$2"; shift 2;;
    --metrics-interval) METRICS_INTERVAL="$2"; shift 2;;
    --links-metrics-dir) LINKS_METRICS_DIR="$2"; shift 2;;
    --links-interval) LINKS_INTERVAL="$2"; shift 2;;
    --queue-analyzer) QUEUE_ANALYZER=1; shift;;
    --capture-conf) CAPTURE_CONF_REL="$2"; shift 2;;
    --help|-h) usage; exit 0;;
    *) echo "Error: unknown option: $1"; usage; exit 1;;
  esac
done

if [[ "$CAPTURE_CONF_REL" = /* ]]; then
  CAPTURE_CONF_HOST="$CAPTURE_CONF_REL"
else
  CAPTURE_CONF_HOST="$TESTBED_DIR/$CAPTURE_CONF_REL"
fi

# ---- SUDO preflight (1 sola vez en foreground) ----
# Importante: sudo cachea por TTY. Por eso *NO* debemos lanzar capturadores con `setsid`
# (pierden la TTY y sudo -n falla con "a password is required").
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  if ! sudo -n true >/dev/null 2>&1; then
    echo "[SUDO] Se requiere contraseña de sudo (una sola vez) para capturas/OVS/docker..." >&2
    sudo -v
  fi
  # Keepalive: refresca el timestamp para que no expire durante el experimento.
  ( while true; do sudo -n true >/dev/null 2>&1 || true; sleep 50; done ) &
  SUDO_KEEPALIVE_PID=$!
fi

# Decide DOCKER wrapper: si docker funciona sin sudo, úsalo; caso contrario, sudo -n.
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  DOCKER=(docker)
else
  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
  else
    DOCKER=(sudo -n docker)
  fi
fi

unique_append() {
  local -n _arr="$1"; local val="$2"
  for x in "${_arr[@]}"; do [[ "$x" == "$val" ]] && return; done
  _arr+=("$val")
}

if [[ -f "$CAPTURE_CONF_HOST" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    case "$line" in
      IFACE=*) unique_append INTERFACES "${line#IFACE=}";;
      CONTAINER=*) unique_append METRICS_CONTAINERS "${line#CONTAINER=}";;
      LINK=*) unique_append LINKS "${line#LINK=}";;
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

mkdir -p "$LOG_DIR" "$CAPTURE_DIR" "$METRICS_DIR" "$LINKS_METRICS_DIR"
RUN_ID=$(date +"%Y%m%d_%H%M%S")

CAPTURE_PID=""; METRICS_PID=""; OVS_METRICS_PID=""
SIM_STARTED=0; STOPPED_CONTAINERS=0
CTRL_LOG_PID=""; SIM_LOG_PID=""; SUDO_KEEPALIVE_PID=""

stop_process() {
  local pid="$1" label="$2" timeout_s="${3:-10}"
  [[ -z "$pid" ]] && return
  ps -p "$pid" >/dev/null 2>&1 || return
  echo "[$label] Stopping..."
  # NO usamos kill a grupo ("-$pid") porque ya no usamos setsid y podrías matar el shell/TTY.
  kill -TERM "$pid" 2>/dev/null || true
  local waited=0
  while ps -p "$pid" >/dev/null 2>&1; do
    if [[ "$waited" -ge "$timeout_s" ]]; then
      echo "[$label] Forcing stop..."
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited+1))
  done
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  echo "[CLEANUP] Starting cleanup sequence..."
  [[ -n "$CTRL_LOG_PID" ]] && "${KILL[@]}" "$CTRL_LOG_PID" 2>/dev/null || true
  [[ -n "$SIM_LOG_PID"  ]] && "${KILL[@]}" "$SIM_LOG_PID"  2>/dev/null || true

  stop_process "$CAPTURE_PID" "CAPTURE"
  stop_process "$METRICS_PID" "METRICS"
  stop_process "$OVS_METRICS_PID" "OVS"

  if "${DOCKER[@]}" ps -q -f "name=^${CONTAINERNET_CONTAINER_NAME}$" >/dev/null; then
    echo "[CLEANUP] Stopping Containernet container ${CONTAINERNET_CONTAINER_NAME}..."
    "${DOCKER[@]}" stop "${CONTAINERNET_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  if [[ $SIM_STARTED -eq 1 && $STOPPED_CONTAINERS -eq 0 && -x "$STOP_CONTAINERS" ]]; then
    echo "[CLEANUP] Running stop_containers.sh..."
    "$STOP_CONTAINERS" || true
    STOPPED_CONTAINERS=1
  fi

  if "${DOCKER[@]}" ps -q -f "name=^${RYU_CONTAINER_NAME}$" >/dev/null; then
    echo "[CLEANUP] Stopping RYU controller container ${RYU_CONTAINER_NAME}..."
    "${DOCKER[@]}" stop "${RYU_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  [[ -n "${SUDO_KEEPALIVE_PID:-}" ]] && kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
  echo "[CLEANUP] Cleanup sequence finished."
}
trap cleanup EXIT INT TERM

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
      local key="${line%%=*}"; local val="${line#*=}"
      case "$key" in
        NAME) ctrl_name="$val";;
        IMAGE) ctrl_image="$val";;
        COMMAND) ctrl_cmd="$val";;
        VOLUME)
          if [[ $custom_volumes -eq 0 ]]; then ctrl_volumes=(); fi
          custom_volumes=1
          ctrl_volumes+=("$val")
          ;;
      esac
    done < <(HOST_PROJECT_ROOT="$TESTBED_DIR" python3 "$PARSE_CONTROLLER_SCRIPT" "$TOPOLOGY_FILE_HOST")
  fi

  RYU_CONTAINER_NAME="$ctrl_name"; RYU_IMAGE="$ctrl_image"

  {
    echo "[CTRL] Image: $RYU_IMAGE"
    echo "[CTRL] Container name: $RYU_CONTAINER_NAME"
    echo "[CTRL] Command: ${ctrl_cmd:-'(image entrypoint)'}"
  } | tee -a "$log_file"

  local run_cmd=("${DOCKER[@]}" run -d --rm --name "$RYU_CONTAINER_NAME" --net=host -e PYTHONPATH=/app)
  for v in "${ctrl_volumes[@]}"; do run_cmd+=(-v "$v"); done
  run_cmd+=("$RYU_IMAGE")
  if [[ -n "$ctrl_cmd" ]]; then
    # shellcheck disable=SC2206
    local ctrl_cmd_arr=($ctrl_cmd)
    run_cmd+=("${ctrl_cmd_arr[@]}")
  fi
  "${run_cmd[@]}" >>"$log_file" 2>&1

  "${DOCKER[@]}" logs -f "${RYU_CONTAINER_NAME}" >>"$log_file" 2>&1 &
  CTRL_LOG_PID=$!
}

start_simulation() {
  echo "[SIM] Starting Containernet simulation container..."
  SIM_STARTED=1
  local log_file="${LOG_DIR}/topology_${RUN_ID}.log"

  {
    echo "[SIM] Image: $CONTAINERNET_IMAGE"
    echo "[SIM] Container name: $CONTAINERNET_CONTAINER_NAME"
    echo "[SIM] Topology script: $TOPOLOGY_SCRIPT_REL"
    echo "[SIM] TOPOLOGY_FILE: $TOPOLOGY_FILE_IN_CONTAINER"
    echo "[SIM] NOTE: --duration in this script controls *capture wall time*;"
    echo "      make sure topology_test.py uses that same duration (via conf/env/args)."
  } | tee -a "$log_file"

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

  "${DOCKER[@]}" logs -f "${CONTAINERNET_CONTAINER_NAME}" >>"$log_file" 2>&1 &
  SIM_LOG_PID=$!
}

start_captures() {
  echo "[CAPTURE] Starting captures on interfaces: ${INTERFACES[*]}"
  local cmd=(sudo -n "$CAPTURE_SCRIPT")
  cmd+=("${INTERFACES[@]}")
  cmd+=(-o "$CAPTURE_DIR")
  [[ "$CAPTURE_SNAPLEN" -gt 0 ]] && cmd+=(-s "$CAPTURE_SNAPLEN")
  local log_file="${LOG_DIR}/capture_${RUN_ID}.log"
  # Importante: NO usar setsid (pierde TTY y sudo -n falla si sudo cachea por TTY)
  "${cmd[@]}" >"$log_file" 2>&1 &
  CAPTURE_PID=$!
  echo "[CAPTURE] PID $CAPTURE_PID (log: $log_file)"
}

start_metrics() {
  [[ ${#METRICS_CONTAINERS[@]} -eq 0 ]] && return
  echo "[METRICS] Starting capture for containers: ${METRICS_CONTAINERS[*]}"
  local cmd=(sudo -n "$METRICS_SCRIPT")
  cmd+=("${METRICS_CONTAINERS[@]}")
  cmd+=(-o "$METRICS_DIR" -i "$METRICS_INTERVAL")
  local log_file="${LOG_DIR}/metrics_${RUN_ID}.log"
  "${cmd[@]}" >"$log_file" 2>&1 &
  METRICS_PID=$!
  echo "[METRICS] PID $METRICS_PID (log: $log_file)"
}

start_ovs_metrics() {
  [[ ${#LINKS[@]} -eq 0 ]] && return
  echo "[OVS] Starting link metrics capture for links: ${LINKS[*]}"
  local cmd=(sudo -n "$OVS_METRICS_SCRIPT")
  cmd+=("${LINKS[@]}")
  cmd+=(-o "$LINKS_METRICS_DIR" -i "$LINKS_INTERVAL")
  [[ "$QUEUE_ANALYZER" -eq 1 ]] && cmd+=(--queue-analyzer)
  local log_file="${LOG_DIR}/ovs_metrics_${RUN_ID}.log"
  "${cmd[@]}" >"$log_file" 2>&1 &
  OVS_METRICS_PID=$!
  echo "[OVS] PID $OVS_METRICS_PID (log: $log_file)"
}

echo "========== AUTOMATED RUN (v2) =========="
echo "Interfaces to capture: ${INTERFACES[*]}"
echo "Target duration:       ${TOTAL_DURATION}s"
echo "Startup buffer:        ${STARTUP_BUFFER}s"
echo "Extra wall margin:     ${MAX_EXTRA_WALL}s"
echo "======================================="

cd "$TESTBED_DIR"

start_controller

# Start capture jobs EARLY so they can attach as soon as resources appear.
start_captures
start_metrics
start_ovs_metrics

start_simulation

MAX_WALL=$((TOTAL_DURATION + STARTUP_BUFFER + MAX_EXTRA_WALL))
echo "[MAIN] Waiting for topology container to finish (max wall ${MAX_WALL}s)..."

if timeout "${MAX_WALL}" "${DOCKER[@]}" wait "${CONTAINERNET_CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[MAIN] Topology container finished. Stopping captures..."
else
  echo "[MAIN] Timeout waiting for topology; stopping container + captures..."
  "${DOCKER[@]}" stop "${CONTAINERNET_CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

stop_process "$CAPTURE_PID" "CAPTURE"
stop_process "$METRICS_PID" "METRICS"
stop_process "$OVS_METRICS_PID" "OVS"

echo "[MAIN] Captures stopped. Running optional teardown..."
if [[ -x "$STOP_CONTAINERS" ]]; then
  "$STOP_CONTAINERS" || true
  STOPPED_CONTAINERS=1
fi

if "${DOCKER[@]}" ps -q -f "name=^${RYU_CONTAINER_NAME}$" >/dev/null; then
  "${DOCKER[@]}" stop "${RYU_CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "[MAIN] Run completed. Check logs in ${LOG_DIR}/"
