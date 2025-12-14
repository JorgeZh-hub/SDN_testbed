#!/usr/bin/env bash
set -euo pipefail

#############################################
# Docker helper                             #
#############################################

# Use sudo docker by default; remove 'sudo' if your user is in the docker group.
DOCKER=(sudo docker)

# Image to use for Mininet/Containernet cleanup
CONTAINERNET_IMAGE="${CONTAINERNET_IMAGE:-containernet_docker}"

#############################################
# Stop and remove experiment containers     #
#############################################

echo "[STOP] Stopping Docker containers..."
# WARNING: this stops ALL containers. If you want to limit it (e.g. by name prefix),
# adjust the --filter line accordingly.
"${DOCKER[@]}" ps -a --format "{{.ID}}" | xargs -r "${DOCKER[@]}" stop

echo "[STOP] Removing stopped containers..."
"${DOCKER[@]}" ps -a --format "{{.ID}}" | xargs -r "${DOCKER[@]}" rm

echo "[STOP] Containers stopped and removed."

#############################################
# Mininet / OVS cleanup via containernet    #
#############################################

echo "[CLEANUP] Running Mininet/OVS cleanup using image: ${CONTAINERNET_IMAGE} ..."

# We run a short-lived Containernet container with host namespaces
# so that mininet.clean.cleanup() operates on the host network and OVS.
"${DOCKER[@]}" run --rm \
  --name containernet-clean \
  --privileged \
  --net=host \
  --pid=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /var/run/openvswitch:/var/run/openvswitch \
  "${CONTAINERNET_IMAGE}" \
  python3 -c "from mininet.clean import cleanup; cleanup()"

echo "[CLEANUP] Mininet/OVS cleanup completed."
