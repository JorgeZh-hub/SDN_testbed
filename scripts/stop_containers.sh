#!/usr/bin/env bash
set -e

DOCKER=(sudo docker)

echo "Stopping Mininet containers..."
"${DOCKER[@]}" ps -a --filter "name=" --format "{{.ID}}" | xargs -r "${DOCKER[@]}" stop

echo "Removing Mininet containers..."
"${DOCKER[@]}" ps -a --filter "name=" --format "{{.ID}}" | xargs -r "${DOCKER[@]}" rm

echo "Containers stopped and removed."

#Limpiar mininet 
sudo PYTHONPATH=/home/tesis/Docs_JZ/containernet python3 -c "from mininet.clean import cleanup; cleanup()"
