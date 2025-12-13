#!/bin/bash

usage() {
    echo "Usage: $0 --ip <server_ip> --path <stream_name> [--port 8890] [--duration seconds]"
    echo "Example: $0 --ip 192.168.1.10 --path mivideo --port 8890 --duration 60"
    exit 1
}

server_ip=""
stream_path=""
port="8890" # Default MediaMTX SRT port
duration=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ip) server_ip="$2"; shift ;;
        --path) stream_path="$2"; shift ;;
        --port) port="$2"; shift ;;
        --duration) duration="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$server_ip" || -z "$stream_path" ]]; then
    usage
fi

start_time=$(date +%s)

# Build SRT URL
srt_url="srt://${server_ip}:${port}?mode=caller&streamid=read:${stream_path}"

while true; do
    echo "Connecting over SRT to ${srt_url}"

    ffmpeg -v error \
      -i "$srt_url" \
      -f null - &

    ffmpeg_pid=$!

    while kill -0 $ffmpeg_pid 2>/dev/null; do
        sleep 1
        if [[ $duration -gt 0 ]]; then
            elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -ge $duration ]]; then
                echo "Maximum receive duration reached (${duration}s). Stopping."
                kill $ffmpeg_pid
                wait $ffmpeg_pid 2>/dev/null
                exit 0
            fi
        fi
    done

    echo "SRT connection lost or stream ended. Reconnecting..."
    sleep 1
done
