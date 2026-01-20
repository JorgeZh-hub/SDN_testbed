#!/usr/bin/env bash
set -euo pipefail

DOCKER=(sudo docker)
CONTAINERNET_IMAGE="${CONTAINERNET_IMAGE:-containernet_docker}"

CTRL_NAME="ryu-ctrl"
SIM_NAME="containernet-sim"
PREFIX="mn."
DO_MN_CLEAN=0
DRY_RUN=0

escape_regex() {
  # escapa regex para grep -E
  sed 's/[][(){}.^$?+*|\\/]/\\&/g' <<<"$1"
}

usage() {
  cat <<EOF
Usage: $0 [options]

Stops/removes ONLY containers from this testbed:
  - controller (default: ryu-ctrl)
  - simulation container (default: containernet-sim)
  - containers with prefix (default: mn.)

Options:
  --controller <name>     Controller container name
  --sim <name>            Simulation container name
  --prefix <prefix>       Prefix for Mininet/Containernet host containers (e.g., mn.)
  --mn-clean              Also run Mininet/OVS cleanup (mininet.clean.cleanup)
  --dry-run               Print what would be stopped/removed
  -h, --help              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --controller) CTRL_NAME="$2"; shift 2 ;;
    --sim)        SIM_NAME="$2"; shift 2 ;;
    --prefix)     PREFIX="$2"; shift 2 ;;
    --mn-clean)   DO_MN_CLEAN=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# 1) Containers by exact name
ids_exact=$("${DOCKER[@]}" ps -aq -f "name=^${CTRL_NAME}$" -f "name=^${SIM_NAME}$" || true)

# 2) Containers by prefix (mn.*)
prefix_re="^$(escape_regex "$PREFIX")"
ids_prefix=$("${DOCKER[@]}" ps -a --format '{{.ID}} {{.Names}}' \
  | awk -v re="$prefix_re" '$2 ~ re {print $1}' || true)

# Merge IDs (unique)
ids_all="$(printf "%s\n%s\n" "$ids_exact" "$ids_prefix" | sed '/^$/d' | sort -u)"

if [[ -z "${ids_all}" ]]; then
  echo "[STOP] No matching containers found (controller=${CTRL_NAME}, sim=${SIM_NAME}, prefix=${PREFIX})."
else
  echo "[STOP] Matching containers:"
  "${DOCKER[@]}" ps -a --format '  {{.ID}}  {{.Names}}' \
    | grep -Ff <(printf "%s\n" "$ids_all") || true

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Not stopping/removing anything."
  else
    echo "[STOP] Stopping..."
    printf "%s\n" "$ids_all" | xargs -r "${DOCKER[@]}" stop

    echo "[STOP] Removing..."
    printf "%s\n" "$ids_all" | xargs -r "${DOCKER[@]}" rm
  fi
fi

# Optional: Mininet/OVS cleanup (can affect other Mininet experiments)
if [[ $DO_MN_CLEAN -eq 1 ]]; then
  echo "[CLEANUP] Running Mininet/OVS cleanup using image: ${CONTAINERNET_IMAGE} ..."
  "${DOCKER[@]}" run --rm \
    --name containernet-clean \
    --privileged \
    --net=host \
    --pid=host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /var/run/openvswitch:/var/run/openvswitch \
    --entrypoint python3 \
    "${CONTAINERNET_IMAGE}" \
    -c "from mininet.clean import cleanup; cleanup()"
fi
