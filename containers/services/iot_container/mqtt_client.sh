#!/usr/bin/env bash

usage() {
  echo "Usage:"
  echo "  $0 -r|-p -b <broker_ip> -t <topic> [-w <kbps>] [-q <qos>] [-T <seg>] [-c <clientes>] [-S <bytes>] [-F <inflight>]"
  echo
  echo "  -r            modo recibir (sub)"
  echo "  -p            modo publicar (pub)"
  echo "  -b <ip>       broker IP or hostname"
  echo "  -t <topic>    MQTT topic"
  echo "  -w <kbps>     target TOTAL bandwidth (approx) in kbps"
  echo "  -q <qos>      QoS (0, 1 or 2). Default: 0"
  echo "  -T <seg>      runtime in seconds (if omitted, runs indefinitely)"
  echo "  -c <num>      number of clients (default: 1)"
  echo "  -S <bytes>    payload size in bytes (default: 100)"
  echo "  -F <num>      inflight for QoS 1/2 (default: 1; try 50-200 for higher throughput)"
  exit 1
}

MODE=""
BROKER=""
TOPIC=""
BANDWIDTH=""   # kbps total
QOS=0
TIME_SECS=""
CLIENTS=1
SIZE=100       # bytes
INFLIGHT=1
PID_CHILD=""

trap 'echo "Interrupted, killing emqtt_bench..."; [ -n "$PID_CHILD" ] && kill "$PID_CHILD" 2>/dev/null; exit 130' INT

while getopts "rpb:t:w:q:T:c:S:F:" opt; do
  case "$opt" in
    r) MODE="sub" ;;
    p) MODE="pub" ;;
    b) BROKER="$OPTARG" ;;
    t) TOPIC="$OPTARG" ;;
    w) BANDWIDTH="$OPTARG" ;;
    q) QOS="$OPTARG" ;;
    T) TIME_SECS="$OPTARG" ;;
    c) CLIENTS="$OPTARG" ;;
    S) SIZE="$OPTARG" ;;
    F) INFLIGHT="$OPTARG" ;;
    *) usage ;;
  esac
done

if [ -z "$MODE" ] || [ -z "$BROKER" ] || [ -z "$TOPIC" ]; then
  echo "Missing required parameters."
  usage
fi

if ! command -v emqtt_bench >/dev/null 2>&1; then
  echo "ERROR: emqtt_bench is not in PATH."
  exit 1
fi

CMD=(emqtt_bench "$MODE" -h "$BROKER" -t "$TOPIC" -q "$QOS" -c "$CLIENTS")

# For QoS1/2, allow more in-flight messages
if [ "$QOS" -ge 1 ]; then
  CMD+=(-F "$INFLIGHT")
fi

# If publisher and bandwidth is set, compute I_ms
if [ "$MODE" = "pub" ] && [ -n "$BANDWIDTH" ]; then
  # I_ms = c * S * 8 / B_total
  I_MS=$(awk -v B="$BANDWIDTH" -v S="$SIZE" -v C="$CLIENTS" 'BEGIN{
      if (B <= 0) { print 1000; exit }
      val = (C*S*8)/B;
      if (val < 1) val = 1;  # lower bound: 1 ms
      printf("%d", val)
  }')

  # Theoretical max throughput with I_ms=1:
  MAX_B=$(awk -v S="$SIZE" -v C="$CLIENTS" 'BEGIN{ print C*S*8 }')

  if [ "$I_MS" -eq 1 ] && [ "$(echo "$BANDWIDTH > $MAX_B" | bc)" -eq 1 ] 2>/dev/null; then
    echo "Warning: requested BW (${BANDWIDTH} kbps) exceeds theoretical max with I=1ms."
    echo "         Approx max with c=${CLIENTS}, S=${SIZE} bytes => ${MAX_B} kbps"
  fi

  CMD+=(-s "$SIZE" -I "$I_MS")
fi

echo "Comando final: ${CMD[*]}"

# ================== No timeout wrapper ==================

if [ -n "$TIME_SECS" ]; then
  echo "Running for ${TIME_SECS}s..."

  START_TS=$(date +%s)

  "${CMD[@]}" &
  PID_CHILD=$!

  while kill -0 "$PID_CHILD" 2>/dev/null; do
    sleep 1
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TS))
    if [ "$ELAPSED" -ge "$TIME_SECS" ]; then
      echo "Maximum duration reached (${TIME_SECS}s). Stopping emqtt_bench..."
      kill "$PID_CHILD" 2>/dev/null
      wait "$PID_CHILD" 2>/dev/null
      exit 0
    fi
  done

  wait "$PID_CHILD"
  exit $?
else
  echo "Running without time limit (Ctrl+C to stop)..."
  "${CMD[@]}"
fi
