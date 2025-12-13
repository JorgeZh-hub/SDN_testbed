#!/bin/bash

usage() {
    echo "Usage: $0 --ip <server_ip> --path <server_path> [--duration seconds]"
    echo "Example: $0 --ip 10.0.0.1 --path mivideo --duration 60"
    exit 1
}

server_ip=""
stream_path=""
duration=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ip) server_ip="$2"; shift ;;
        --path) stream_path="$2"; shift ;;
        --duration) duration="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$server_ip" || -z "$stream_path" ]]; then
    usage
fi

start_time=$(date +%s)

while true; do
    echo "Receiving HLS from http://${server_ip}:8888/${stream_path}/index.m3u8"

    ffmpeg -loglevel error -fflags nobuffer \
      -i "http://${server_ip}:8888/${stream_path}/index.m3u8" \
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

    echo "ffmpeg HLS process stopped. Restarting reception..."
    sleep 1
done
