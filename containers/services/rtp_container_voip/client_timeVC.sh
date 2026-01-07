#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [SERVER_IP] [CLIENT_IP] [--duration SECONDS]"
    exit 1
fi

server_ip=$1
client_ip=$2
scenario_dir=${SIPP_SCENARIO_DIR:-/data/voip}
xml_file="${scenario_dir}/uac_pcapVC.xml"
duration=0  # 0 = run indefinitely

shift 2
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --duration)
            duration="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ! -f "$xml_file" ]; then
    echo "Scenario file not found: $xml_file"
    exit 1
fi

start_time=$(date +%s)

echo "Starting SIPp client with audio/video towards $server_ip from $client_ip"
if [[ "$duration" -gt 0 ]]; then
    echo "Running for $duration seconds..."
else
    echo "Running until interrupted..."
fi

while true; do
    if [[ $duration -gt 0 ]]; then
        elapsed=$(( $(date +%s) - start_time ))
        if [[ $elapsed -ge $duration ]]; then
            echo "Maximum duration reached ($duration s). Exiting."
            exit 0
        fi
    fi

    echo "Starting SIP call with media..."
    sipp -sf "$xml_file" "$server_ip" \
         -i "$client_ip" -mi "$client_ip" \
         -t u1 -r 10 -l 10 \
         -recv_timeout 10000 &

    sipp_pid=$!

    while kill -0 $sipp_pid 2>/dev/null; do
        sleep 1
        if [[ $duration -gt 0 ]]; then
            elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -ge $duration ]]; then
                echo "Duration reached. Stopping active call..."
                kill $sipp_pid
                wait $sipp_pid 2>/dev/null
                exit 0
            fi
        fi
    done

    echo "Call finished. Restarting in 1 second..."
    sleep 1
done
