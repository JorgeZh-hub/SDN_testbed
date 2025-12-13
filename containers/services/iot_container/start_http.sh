#!/bin/bash
set -e

WORKERS=${WORKERS:-2}
THREADS=${THREADS:-4}
PORT=${PORT:-5000}

usage() {
    echo "Usage: $0 [-w workers] [-T threads] [-P port]"
    echo "Defaults: workers=${WORKERS}, threads=${THREADS}, port=${PORT}"
    exit 1
}

while getopts "w:T:P:h" opt; do
    case "$opt" in
        w) WORKERS="$OPTARG" ;;
        T) THREADS="$OPTARG" ;;
        P) PORT="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "Starting HTTP with gunicorn (workers=$WORKERS threads=$THREADS) on port $PORT"
exec gunicorn -w "$WORKERS" -k gthread --threads "$THREADS" -b "0.0.0.0:${PORT}" server:app
