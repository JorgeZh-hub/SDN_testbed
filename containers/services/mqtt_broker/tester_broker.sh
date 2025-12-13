#!/bin/bash

# Interval between checks (seconds)
INTERVAL=10

# Counter of time without connections
no_conn_duration=0

# Threshold (seconds) without connections before restarting Mosquitto
THRESHOLD=3

echo "Starting port 1883 monitoring..."

while true; do
    # Check for established connections on port 1883
    conn_count=$(ss -ant sport = :1883 state established | grep -v "Recv-Q" | wc -l)

    if [ "$conn_count" -eq 0 ]; then
        no_conn_duration=$((no_conn_duration + INTERVAL))
        echo "No active connections on port 1883 for ${no_conn_duration}s."
    else
        echo "Active connections detected: $conn_count"
        no_conn_duration=0
    fi

    if [ "$no_conn_duration" -ge "$THRESHOLD" ]; then
        echo "Restarting Mosquitto after ${no_conn_duration}s without connections..."

        # Find and stop Mosquitto
        MOSQ_PID=$(pgrep -f "/usr/sbin/mosquitto")
        if [ -n "$MOSQ_PID" ]; then
            kill -9 "$MOSQ_PID"
            sleep 1
        fi

        # Restart Mosquitto
        /usr/sbin/mosquitto -c /etc/mosquitto/mosquitto.conf -v &

        echo "Mosquitto restarted."

        # Reset counter to avoid repeated restarts
        no_conn_duration=0
    fi

    sleep "$INTERVAL"
done
