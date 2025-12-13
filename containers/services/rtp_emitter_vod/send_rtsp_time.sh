#!/bin/bash

usage() {
    echo "Usage: $0 --ip <server_ip> --path <server_path> [--video video_path] [--duration seconds]"
    echo "Example: $0 --ip 192.168.1.10 --path mivideo --video file.mp4 --duration 60"
    echo "Available paths: mivideo, mivideo2"
    exit 1
}

video_file=""
server_ip=""
stream_path=""
duration=0  # default: run indefinitely

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video) video_file="$2"; shift ;;
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

if [[ -z "$video_file" ]]; then
    echo "A video file is required. Provide it with --video."
    usage
fi

start_time=$(date +%s)

while true; do
    echo "Streaming $video_file to rtsp://${server_ip}:8554/${stream_path}"

    ffmpeg -re -stream_loop -1 -i "$video_file" -c copy -f rtsp "rtsp://${server_ip}:8554/${stream_path}" &
    ffmpeg_pid=$!

    while kill -0 $ffmpeg_pid 2>/dev/null; do
        sleep 1
        if [[ $duration -gt 0 ]]; then
            elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -ge $duration ]]; then
                echo "Maximum streaming duration reached (${duration}s). Stopping."
                kill $ffmpeg_pid
                wait $ffmpeg_pid 2>/dev/null
                exit 0
            fi
        fi
    done

    echo "ffmpeg stopped. Restarting stream..."
    sleep 1
done
