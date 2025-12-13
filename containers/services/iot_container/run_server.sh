#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: provide the Python script path as an argument."
    echo "Usage: $0 /path/to/server.py"
    exit 1
fi

SCRIPT_PATH="$1"
PORT=${PORT_OVERRIDE:-8683}

extraer_pids_por_puerto() {
    ss -tulnp 2>/dev/null | grep "$PORT" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u
}

while true; do
    echo "Starting server: $SCRIPT_PATH"
    python3 "$SCRIPT_PATH" &
    SERVER_PID=$!

    sleep 120

    echo "Restarting server. Stopping PID ${SERVER_PID}..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null

    echo "Checking for stray processes on port $PORT"
    PIDS=$(extraer_pids_por_puerto)
    if [ -n "$PIDS" ]; then
        for PID in $PIDS; do
            echo "Process on port $PORT (PID $PID) detected, terminating..."
            kill "$PID" 2>/dev/null
        done
    else
        echo "Port $PORT is clear."
    fi
done
