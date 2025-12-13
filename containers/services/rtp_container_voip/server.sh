#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [SERVER_IP]"
    exit 1
fi

server_ip=$1

# Run SIPp as server (uas) with RTP echo
sipp -sn uas -rtp_echo -i "$server_ip" -t u1 -p 5060
