#!/bin/bash

usage() {
    echo "Usage: $0 --ip <server_ip> --video <video_name> [--duration seconds]"
    echo "Example: $0 --ip 192.168.1.10 --video demo.mp4 --duration 60"
    exit 1
}

server_ip=""
video_name=""
duration=0  # default: no limit

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ip) server_ip="$2"; shift ;;
        --video) video_name="$2"; shift ;;
        --duration) duration="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$server_ip" || -z "$video_name" ]]; then
    usage
fi

temp_video="./temp_video.mp4"
start_time=$(date +%s)

while true; do
    echo "Requesting video http://${server_ip}/videos/${video_name}"

    curl -s -o "$temp_video" "http://${server_ip}/videos/${video_name}" &
    curl_pid=$!

    while kill -0 $curl_pid 2>/dev/null; do
        sleep 1
        if [[ $duration -gt 0 ]]; then
            elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -ge $duration ]]; then
                echo "Duration reached (${duration}s). Stopping download..."
                kill $curl_pid
                wait $curl_pid 2>/dev/null
                rm -f "$temp_video"
                exit 0
            fi
        fi
    done

    echo "Download finished early. Restarting..."
    sleep 1
done
