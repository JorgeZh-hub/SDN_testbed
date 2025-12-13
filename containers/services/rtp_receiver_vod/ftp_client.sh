#!/usr/bin/env bash

usage() {
  echo "Usage:"
  echo "  $0 -s <server_ip> -f <remote_path> -U <user> -W <pass> [-P <port>] [-T <seconds>] [-l <local_file>]"
  echo
  echo "Examples:"
  echo "  $0 -s 203.0.113.7 -f /bigfile.bin -U ftpuser -W ftppass -T 60"
  echo "  $0 -s 203.0.113.7 -f /bigfile.bin -U ftpuser -W ftppass      # run until interrupted"
  exit 1
}

SERVER=""
PORT=21
REMOTE_FILE=""
USER_FTP=""
PASS_FTP=""
TIME_SECS=0          # 0 = no limit
LOCAL_FILE="/dev/null"

trap 'echo "Interrupted, exiting..."; exit 130' INT

while getopts "s:P:f:U:W:T:l:" opt; do
  case "$opt" in
    s) SERVER="$OPTARG" ;;
    P) PORT="$OPTARG" ;;
    f) REMOTE_FILE="$OPTARG" ;;
    U) USER_FTP="$OPTARG" ;;
    W) PASS_FTP="$OPTARG" ;;
    T) TIME_SECS="$OPTARG" ;;
    l) LOCAL_FILE="$OPTARG" ;;
    *) usage ;;
  esac
done

if [ -z "$SERVER" ] || [ -z "$REMOTE_FILE" ] || [ -z "$USER_FTP" ] || [ -z "$PASS_FTP" ]; then
  echo "Missing required parameters."
  usage
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is not installed."
  exit 1
fi

FTP_URL="ftp://${USER_FTP}:${PASS_FTP}@${SERVER}:${PORT}${REMOTE_FILE}"

echo "Generating FTP traffic (single client)"
echo "Server: ${SERVER}:${PORT}"
echo "Remote file: ${REMOTE_FILE}"
echo "Local destination: ${LOCAL_FILE}"
[ "$TIME_SECS" -gt 0 ] && echo "Max duration: ${TIME_SECS}s" || echo "Duration: unlimited (Ctrl+C to stop)"

START_TIME=$(date +%s)
if [ "$TIME_SECS" -gt 0 ]; then
  END_TIME=$((START_TIME + TIME_SECS))
else
  END_TIME=0
fi

while true; do
  if [ "$END_TIME" -ne 0 ]; then
    NOW=$(date +%s)
    if [ "$NOW" -ge "$END_TIME" ]; then
      echo "Maximum duration reached. Stopping downloads."
      break
    fi
    REMAINING_TIME=$((END_TIME - NOW))
  else
    REMAINING_TIME=0
  fi

  if [ "$REMAINING_TIME" -gt 0 ]; then
    curl -s --max-time "$REMAINING_TIME" --ftp-pasv "$FTP_URL" -o "$LOCAL_FILE"
    rc=$?
    if [ "$rc" -eq 28 ]; then  # CURLE_OPERATION_TIMEDOUT
      echo "Maximum time reached during download. Stopping."
      break
    fi
  else
    curl -s --ftp-pasv "$FTP_URL" -o "$LOCAL_FILE"
    rc=$?
  fi

  if [ "$rc" -ne 0 ]; then
    echo "curl returned error (rc=$rc), stopping client."
    break
  fi
done

echo "FTP client finished."
exit 0
