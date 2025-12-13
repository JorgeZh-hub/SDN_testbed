#!/usr/bin/env bash

############################################
#   OVS LINK METRICS CAPTURE
#
#   Usage:
#     ./capture_link_metrics.sh "sw1-if1, sw2-if2, cap_mbps" [...] -t 60 -o dir -i 1 [--queue-analyzer]
#
#   Example:
#     ./capture_link_metrics.sh \
#       "switch14-eth1, switch15-eth2, 30" \
#       "switch1-eth1, switch5-eth2, 50" \
#       -t 300 -o ./link_metrics -i 1 --queue-analyzer
#
#   Each link is given as:
#     "<sw1>-<if1>, <sw2>-<if2>, <capacity_mbps>"
#   Capacity (Mbps) is used to compute utilization%.
#
#   Output:
#     - One CSV per link: <OUTPUT_DIR>/<link>.csv
#       Columns: timestamp,mbps,rx_drop,tx_drop,total_drop,avg_queue_bytes,util_pct
#
#     - If --queue-analyzer is enabled:
#       One CSV per switch+queue: <OUTPUT_DIR>/<switch>_queue<id>.csv
#       Columns: timestamp,port,queue_id,tx_bytes,tx_packets,tx_errors
############################################

OVS_VSCTL=(sudo ovs-vsctl)
OVS_OFCTL=(sudo ovs-ofctl)
TC=(sudo tc)

DURATION=60
OUTPUT_DIR="./link_metrics"
INTERVAL=1
QUEUE_ANALYZER=0
LINKS=()
MAX_OFPORT_ATTEMPTS=60
OFPORT_SLEEP=2

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--time)
            DURATION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        --queue-analyzer)
            QUEUE_ANALYZER=1
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 link1 link2 ... -t <seconds> -o <dir> [-i interval] [--queue-analyzer]"
            exit 1
            ;;
        *)
            LINKS+=("$1")
            shift
            ;;
    esac
done

if [ ${#LINKS[@]} -eq 0 ]; then
    echo "Usage: $0 link1 link2 ... -t <seconds> -o <dir> [-i interval] [--queue-analyzer]"
    echo "Example: $0 switch14-eth1-switch15-eth2 -t 300 -o ./link_metrics -i 1"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---------- Helper: get OVS port counters for one port ----------
# Output: "rx_bytes tx_bytes rx_drop tx_drop"
get_port_counters() {
    local br="$1"
    local ofport="$2"

    "${OVS_OFCTL[@]}" -O OpenFlow13 dump-ports "$br" "$ofport" 2>/dev/null | \
    awk '
        /rx pkts=/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^bytes=/) {
                    rb = $i; gsub("bytes=", "", rb); gsub(",", "", rb)
                }
                if ($i ~ /^drop=/) {
                    rd = $i; gsub("drop=", "", rd); gsub(",", "", rd)
                }
            }
        }
        /tx pkts=/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^bytes=/) {
                    tb = $i; gsub("bytes=", "", tb); gsub(",", "", tb)
                }
                if ($i ~ /^drop=/) {
                    td = $i; gsub("drop=", "", td); gsub(",", "", td)
                }
            }
        }
        END {
            if (rb == "") rb = 0;
            if (tb == "") tb = 0;
            if (rd == "") rd = 0;
            if (td == "") td = 0;
            print rb, tb, rd, td
        }'
}

# ---------- Helper: get backlog (queue size) in bytes for an interface ----------
# Uses: tc -s qdisc show dev <iface>  (backlog Xb ...)
get_iface_backlog_bytes() {
    local iface="$1"
    local line
    local val

    line=$("${TC[@]}" -s qdisc show dev "$iface" 2>/dev/null | grep "backlog" | head -n1)
    if [ -z "$line" ]; then
        echo 0
        return
    fi

    val=$(echo "$line" | awk '{print $2}')   # e.g. "0b" or "12Kb" or "1234b"
    # Remove trailing "b"
    val=${val%b}

    # Now val might be "0", "1234", "12K", "1M"
    local last=${val: -1}
    local num unit factor

    if [[ "$last" =~ [KkMm] ]]; then
        unit="$last"
        num="${val%$last}"
    else
        unit=""
        num="$val"
    fi

    case "$unit" in
        K|k)
            factor=1000
            ;;
        M|m)
            factor=1000000
            ;;
        *)
            factor=1
            ;;
    esac

    if [ -z "$num" ]; then
        echo 0
        return
    fi

    awk -v n="$num" -v f="$factor" 'BEGIN { printf "%d\n", n * f }'
}

# ---------- Helper: wait for a valid ofport for an interface ----------
wait_for_ofport() {
    local iface="$1"
    local attempts=0
    while [ "$attempts" -lt "$MAX_OFPORT_ATTEMPTS" ]; do
        local ofp
        ofp=$("${OVS_VSCTL[@]}" --if-exists get Interface "$iface" ofport 2>/dev/null | tr -d '[:space:]')
        if [ -n "$ofp" ] && [ "$ofp" != "-1" ]; then
            echo "$ofp"
            return 0
        fi
        attempts=$((attempts + 1))
        sleep "$OFPORT_SLEEP"
    done
    return 1
}

# ---------- Helpers ----------
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"   # trim leading
    s="${s%"${s##*[![:space:]]}"}"   # trim trailing
    printf "%s" "$s"
}

# ---------- Helper: unique switches set from links ----------
SWITCHES=()

add_switch_if_new() {
    local sw="$1"
    for s in "${SWITCHES[@]}"; do
        if [ "$s" = "$sw" ]; then
            return
        fi
    done
    SWITCHES+=("$sw")
}

# ---------- Parse links into endpoints ----------
# Arrays per index i:
#   LINK_LABEL[i]
#   A_SWITCH[i], A_IFACE[i], A_OFPORT[i]
#   B_SWITCH[i], B_IFACE[i], B_OFPORT[i]
#   LINK_CAP_MBPS[i]

declare -a LINK_LABEL
declare -a A_SWITCH A_IFACE A_OFPORT
declare -a B_SWITCH B_IFACE B_OFPORT
declare -a LINK_CAP_MBPS
declare -a PREV_TS

idx=0
for link in "${LINKS[@]}"; do
    cap_mbps=0
    raw_sw1_if1=""
    raw_sw2_if2=""

    if [[ "$link" == *,* ]]; then
        IFS=',' read -r part1 part2 part3 <<< "$link"
        raw_sw1_if1=$(trim "$part1")
        raw_sw2_if2=$(trim "$part2")
        cap_mbps=$(trim "$part3")
        IFS='-' read -r sw1 if1 <<< "$raw_sw1_if1"
        IFS='-' read -r sw2 if2 <<< "$raw_sw2_if2"
    else
        # Backward-compatible format: sw1-if1-sw2-if2
        IFS='-' read -r sw1 if1 sw2 if2 <<< "$link"
        cap_mbps=0
    fi

    if [ -z "$sw1" ] || [ -z "$if1" ] || [ -z "$sw2" ] || [ -z "$if2" ]; then
        echo "Invalid link format: $link"
        echo "Expected: \"sw1-if1, sw2-if2, capacity_mbps\" (e.g., \"switch1-eth1, switch2-eth2, 30\")"
        exit 1
    fi

    local_iface1="${sw1}-${if1}"
    local_iface2="${sw2}-${if2}"

    # Get ofport numbers from OVS
    if ! ofport1=$(wait_for_ofport "$local_iface1"); then
        echo "Could not get ofport for interface $local_iface1 after waiting ${MAX_OFPORT_ATTEMPTS}*${OFPORT_SLEEP}s"
        exit 1
    fi
    if ! ofport2=$(wait_for_ofport "$local_iface2"); then
        echo "Could not get ofport for interface $local_iface2 after waiting ${MAX_OFPORT_ATTEMPTS}*${OFPORT_SLEEP}s"
        exit 1
    fi

    LINK_LABEL[$idx]="${sw1}-${if1}-${sw2}-${if2}"
    LINK_CAP_MBPS[$idx]="$cap_mbps"

    A_SWITCH[$idx]="$sw1"
    A_IFACE[$idx]="$local_iface1"
    A_OFPORT[$idx]="$ofport1"

    B_SWITCH[$idx]="$sw2"
    B_IFACE[$idx]="$local_iface2"
    B_OFPORT[$idx]="$ofport2"

    add_switch_if_new "$sw1"
    add_switch_if_new "$sw2"

    idx=$((idx + 1))
done

LINK_COUNT=${#LINK_LABEL[@]}

# ---------- Prepare CSV files for each link ----------
for ((i=0; i<LINK_COUNT; i++)); do
    csv_file="${OUTPUT_DIR}/${LINK_LABEL[$i]}.csv"
    echo "timestamp,mbps,rx_drop,tx_drop,total_drop,avg_queue_bytes,util_pct" > "$csv_file"
done

# ---------- State: previous counters per endpoint ----------
declare -a PREV_A_RX_BYTES PREV_A_TX_BYTES PREV_A_RX_DROP PREV_A_TX_DROP
declare -a PREV_B_RX_BYTES PREV_B_TX_BYTES PREV_B_RX_DROP PREV_B_TX_DROP
declare -a HAVE_PREV_A HAVE_PREV_B

for ((i=0; i<LINK_COUNT; i++)); do
    PREV_A_RX_BYTES[$i]=""
    PREV_A_TX_BYTES[$i]=""
    PREV_A_RX_DROP[$i]=""
    PREV_A_TX_DROP[$i]=""
    PREV_B_RX_BYTES[$i]=""
    PREV_B_TX_BYTES[$i]=""
    PREV_B_RX_DROP[$i]=""
    PREV_B_TX_DROP[$i]=""
    HAVE_PREV_A[$i]=0
    HAVE_PREV_B[$i]=0
    PREV_TS[$i]=""
done

echo "============================================="
echo "  STARTING OVS LINK METRICS CAPTURE"
echo "  Links:"
for ((i=0; i<LINK_COUNT; i++)); do
    cap="${LINK_CAP_MBPS[$i]}"
    echo "    ${LINK_LABEL[$i]}  (A: ${A_SWITCH[$i]}/${A_IFACE[$i]} port ${A_OFPORT[$i]}, B: ${B_SWITCH[$i]}/${B_IFACE[$i]} port ${B_OFPORT[$i]}, cap=${cap} Mbps)"
done
echo "  Duration:   ${DURATION}s"
echo "  Interval:   ${INTERVAL}s"
echo "  Folder:     $OUTPUT_DIR"
echo "  Queue analyzer: $QUEUE_ANALYZER"
echo "============================================="

END_TIME=$(( $(date +%s) + DURATION ))

# ---------- Main sampling loop ----------
while [ "$(date +%s)" -lt "$END_TIME" ]; do
    TS=$(date +%s)

    # Per-link metrics
    for ((i=0; i<LINK_COUNT; i++)); do
        prev_ts="${PREV_TS[$i]}"
        a_sw="${A_SWITCH[$i]}"
        a_port="${A_OFPORT[$i]}"
        b_sw="${B_SWITCH[$i]}"
        b_port="${B_OFPORT[$i]}"
        cap_mbps="${LINK_CAP_MBPS[$i]}"
        csv_file="${OUTPUT_DIR}/${LINK_LABEL[$i]}.csv"

        # Get current counters for A and B ports
        read a_rx_bytes a_tx_bytes a_rx_drop a_tx_drop < <(get_port_counters "$a_sw" "$a_port")
        read b_rx_bytes b_tx_bytes b_rx_drop b_tx_drop < <(get_port_counters "$b_sw" "$b_port")

        # Compute deltas only if we have previous sample
        if [ "${HAVE_PREV_A[$i]}" -eq 1 ]; then
            d_a_rx_bytes=$((a_rx_bytes - PREV_A_RX_BYTES[$i]))
            d_a_tx_bytes=$((a_tx_bytes - PREV_A_TX_BYTES[$i]))
            d_a_rx_drop=$((a_rx_drop - PREV_A_RX_DROP[$i]))
            d_a_tx_drop=$((a_tx_drop - PREV_A_TX_DROP[$i]))
            [ "$d_a_rx_bytes" -lt 0 ] && d_a_rx_bytes=0
            [ "$d_a_tx_bytes" -lt 0 ] && d_a_tx_bytes=0
            [ "$d_a_rx_drop" -lt 0 ] && d_a_rx_drop=0
            [ "$d_a_tx_drop" -lt 0 ] && d_a_tx_drop=0
        else
            d_a_rx_bytes=0
            d_a_tx_bytes=0
            d_a_rx_drop=0
            d_a_tx_drop=0
        fi

        if [ "${HAVE_PREV_B[$i]}" -eq 1 ]; then
            d_b_rx_bytes=$((b_rx_bytes - PREV_B_RX_BYTES[$i]))
            d_b_tx_bytes=$((b_tx_bytes - PREV_B_TX_BYTES[$i]))
            d_b_rx_drop=$((b_rx_drop - PREV_B_RX_DROP[$i]))
            d_b_tx_drop=$((b_tx_drop - PREV_B_TX_DROP[$i]))
            [ "$d_b_rx_bytes" -lt 0 ] && d_b_rx_bytes=0
            [ "$d_b_tx_bytes" -lt 0 ] && d_b_tx_bytes=0
            [ "$d_b_rx_drop" -lt 0 ] && d_b_rx_drop=0
            [ "$d_b_tx_drop" -lt 0 ] && d_b_tx_drop=0
        else
            d_b_rx_bytes=0
            d_b_tx_bytes=0
            d_b_rx_drop=0
            d_b_tx_drop=0
        fi

        # Total bytes and drops in this interval for the whole link
        total_rx_bytes=$((d_a_rx_bytes + d_b_rx_bytes))
        total_tx_bytes=$((d_a_tx_bytes + d_b_tx_bytes))
        total_rx_drop=$((d_a_rx_drop + d_b_rx_drop))
        total_tx_drop=$((d_a_tx_drop + d_b_tx_drop))
        total_drop=$((total_rx_drop + total_tx_drop))
        total_a_bytes=$((d_a_rx_bytes + d_a_tx_bytes))
        total_b_bytes=$((d_b_rx_bytes + d_b_tx_bytes))

        # Mbps (sum of RX+TX in the link)
        # Mb/s = bytes * 8 / (interval * 1e6)
        # Use actual elapsed time to avoid spikes from scheduling jitter.
        dt=$INTERVAL
        if [ -n "$prev_ts" ]; then
            dt=$((TS - prev_ts))
            [ "$dt" -le 0 ] && dt=$INTERVAL
        fi
        # We average both endpoints to avoid double-counting the same traffic.
        mbps=$(LC_NUMERIC=C awk -v a="$total_a_bytes" -v b="$total_b_bytes" -v dt="$dt" '
            BEGIN {
                if (dt <= 0) {
                    printf "0.000\n";
                    exit;
                }
                total_bytes = (a + b) / 2.0;
                mbps = (total_bytes * 8.0) / (dt * 1000000.0);
                printf "%.3f\n", mbps;
            }')

        # Average queue backlog in bytes (simple average of both endpoints)
        backlog_a=$(get_iface_backlog_bytes "${A_IFACE[$i]}")
        backlog_b=$(get_iface_backlog_bytes "${B_IFACE[$i]}")
        avg_queue_bytes=$(awk -v a="$backlog_a" -v b="$backlog_b" 'BEGIN { print (a + b) / 2.0 }')

        util_pct=$(LC_NUMERIC=C awk -v mb="$mbps" -v cap="$cap_mbps" '
            BEGIN {
                if (cap + 0 <= 0) {
                    printf "0.000\n";
                    exit;
                }
                printf "%.3f\n", (mb / cap) * 100.0;
            }')

        # Append to CSV
        echo "$TS,$mbps,$total_rx_drop,$total_tx_drop,$total_drop,$avg_queue_bytes,$util_pct" >> "$csv_file"

        # Store current counters as previous
        PREV_A_RX_BYTES[$i]=$a_rx_bytes
        PREV_A_TX_BYTES[$i]=$a_tx_bytes
        PREV_A_RX_DROP[$i]=$a_rx_drop
        PREV_A_TX_DROP[$i]=$a_tx_drop
        PREV_B_RX_BYTES[$i]=$b_rx_bytes
        PREV_B_TX_BYTES[$i]=$b_tx_bytes
        PREV_B_RX_DROP[$i]=$b_rx_drop
        PREV_B_TX_DROP[$i]=$b_tx_drop
        HAVE_PREV_A[$i]=1
        HAVE_PREV_B[$i]=1
        PREV_TS[$i]=$TS
    done

    # Optional: queue analyzer per switch
    if [ "$QUEUE_ANALYZER" -eq 1 ]; then
        for sw in "${SWITCHES[@]}"; do
            "${OVS_OFCTL[@]}" -O OpenFlow13 dump-queue-stats "$sw" 2>/dev/null | \
            awk -v ts="$TS" -v sw="$sw" -v outdir="$OUTPUT_DIR" '
                /port [0-9]+: queue_id [0-9]+:/ {
                    port = ""; qid = "";
                    for (i = 1; i <= NF; i++) {
                        if ($i ~ /^port$/) {
                            port = $(i+1); sub(":", "", port);
                        }
                        if ($i ~ /^queue_id$/) {
                            qid = $(i+1); sub(":", "", qid);
                        }
                    }
                    getline;
                    txb = txp = txe = 0;
                    for (i = 1; i <= NF; i++) {
                        if ($i ~ /^tx_bytes=/) {
                            txb = $i; gsub("tx_bytes=", "", txb); gsub(",", "", txb);
                        }
                        if ($i ~ /^tx_packets=/) {
                            txp = $i; gsub("tx_packets=", "", txp); gsub(",", "", txp);
                        }
                        if ($i ~ /^tx_errors=/) {
                            txe = $i; gsub("tx_errors=", "", txe); gsub(",", "", txe);
                        }
                    }
                    fname = outdir "/" sw "_queue" qid ".csv";
                    if (!(fname in header_written)) {
                        print "timestamp,port,queue_id,tx_bytes,tx_packets,tx_errors" > fname;
                        header_written[fname] = 1;
                    }
                    print ts "," port "," qid "," txb "," txp "," txe >> fname;
                    close(fname);
                }'
        done
    fi

    sleep "$INTERVAL"
done

echo "Link metrics stored in: $OUTPUT_DIR"
echo "============================================="
