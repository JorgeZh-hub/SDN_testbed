#!/bin/bash

# Usage: ./iperf_retry.sh [iperf3 options]
# Example: ./iperf_retry.sh -c 192.51.100.22 -u -b 1M

if ! command -v iperf3 >/dev/null 2>&1; then
    echo "iperf3 is not installed. Aborting."
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 [iperf3 options]"
    echo "Example: $0 -c 192.51.100.22 -u -b 1M"
    exit 1
fi

CMD="iperf3 $*"
echo "Running command: $CMD"

while true; do
    echo "Attempting: $CMD"
    $CMD
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Command executed successfully."
        break
    else
        echo "Command failed (exit code $EXIT_CODE). Retrying in 5 seconds..."
        sleep 5
    fi
done
