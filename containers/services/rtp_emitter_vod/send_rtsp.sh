#!/bin/bash

usage() {
    echo "Usage: $0 --ip <server_ip> --path <server_path> [--video video_path] [--bw kbps]"
    echo "Example: $0 --ip 192.168.1.10 --path mivideo --video file.mp4 --bw 2048"
    echo "Available paths: mivideo, mivideo2"
    exit 1
}

video_file=""
server_ip=""
stream_path=""
bandwidth_kbps=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video) video_file="$2"; shift ;;
        --ip) server_ip="$2"; shift ;;
        --path) stream_path="$2"; shift ;;
        --bw) bandwidth_kbps="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$server_ip" || -z "$stream_path" ]]; then
    usage
fi

while true; do
    if [[ -n "$video_file" ]]; then
        echo "Streaming $video_file to rtsp://${server_ip}:8554/${stream_path}"

        if [[ -n "$bandwidth_kbps" ]]; then
            echo "Limiting bitrate to ${bandwidth_kbps} Kbps"
            ffmpeg -re -stream_loop -1 -i "$video_file" \
                -c:v libx264 -b:v "${bandwidth_kbps}k" -maxrate "${bandwidth_kbps}k" -bufsize "$((bandwidth_kbps * 2))k" \
                -f rtsp "rtsp://${server_ip}:8554/${stream_path}"
        else
            echo "Streaming without bitrate cap"
            ffmpeg -re -stream_loop -1 -i "$video_file" -c copy -f rtsp "rtsp://${server_ip}:8554/${stream_path}"
        fi
    else
        echo "A video file is required. Provide it with --video."
        usage
    fi
    echo "ffmpeg stopped, restarting in 1 second..."
    sleep 1
done
