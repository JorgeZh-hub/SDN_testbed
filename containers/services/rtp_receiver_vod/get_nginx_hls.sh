#!/bin/bash

usage() {
    echo "Usage: $0 --ip <server_ip> --path <m3u8_path> [--duration seconds]"
    echo "Example: $0 --ip 192.168.1.10 --path videos/hls/my_video/index.m3u8 --duration 60"
    exit 1
}

server_ip=""
playlist_path=""
duration=0  # default: no limit

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ip) server_ip="$2"; shift ;;
        --path) playlist_path="$2"; shift ;;
        --duration) duration="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$server_ip" || -z "$playlist_path" ]]; then
    usage
fi

start_time=$(date +%s)
base_url="http://${server_ip}/$(dirname "$playlist_path")"

echo "Starting HLS traffic generation against ${server_ip}..."

while true; do
    stop_loop=0

    echo "Fetching playlist: http://${server_ip}/${playlist_path}"
    playlist_content=$(curl -s "http://${server_ip}/${playlist_path}")

    if [[ -z "$playlist_content" ]]; then
        echo "Failed to download the .m3u8 playlist or it is empty."
        sleep 2
        continue
    fi

    # Download each .ts segment listed in the playlist
    while IFS= read -r segment; do
        segment_url="$base_url/$segment"

        curl -s -o /dev/null "$segment_url"

        if [[ $duration -gt 0 ]]; then
            elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -ge $duration ]]; then
                echo "Duration reached (${duration}s). Stopping..."
                stop_loop=1
                break
            fi
        fi
    done < <(echo "$playlist_content" | grep -v "^#" | grep ".ts")

    if [[ $stop_loop -eq 1 ]]; then
        break
    fi

    echo "Playlist completed. Restarting HLS cycle..."
    sleep 1
done

exit 0
