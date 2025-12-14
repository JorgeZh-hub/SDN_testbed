#!/usr/bin/env bash
set -euo pipefail

#############################################
# Docker helper                             #
#############################################

# Use sudo docker by default; change to DOCKER=(docker) if not needed.
DOCKER=(sudo docker)

# Image to use for Mininet/Containernet cleanup
CONTAINERNET_IMAGE="${CONTAINERNET_IMAGE:-containernet_docker}"

#############################################
# Stop and remove experiment containers     #
#############################################

echo "[STOP] Stopping Docker containers..."
# WARNING: this stops ALL containers. If you want to restrict this later,
# adjust the 'docker ps' filter to match only your lab containers.
"${DOCKER[@]}" ps -a --format "{{.ID}}" | xargs -r "${DOCKER[@]}" stop

echo "[STOP] Removing stopped containers..."
"${DOCKER[@]}" ps -a --format "{{.ID}}" | xargs -r "${DOCKER[@]}" rm

echo "[STOP] Containers stopped and removed."

#############################################
# Mininet / OVS cleanup via Containernet    #
#############################################

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

echo "[CLEANUP] Mininet/OVS cleanup completed."
