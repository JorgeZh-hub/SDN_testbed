#!/usr/bin/env bash
set -euo pipefail


############################################
# v2: OVS LINK METRICS CAPTURE
#
# Problema de v1:
#   Para cada link, hace wait_for_ofport ANTES de iniciar el muestreo.
#   Si 1 interfaz/ofport tarda mucho, bloquea TODO el script y terminas
#   capturando solo los "últimos segundos".
#
# Solución v2:
#   - Resuelve ofport de forma "lazy" por link.
#   - Si un ofport aún no existe, ese link se registra con 0 y se reintenta
#     en la siguiente iteración, sin bloquear a los demás links.
############################################

## ---- SUDO helper ----
# Si se ejecuta como root, no uses sudo. Si no, intenta sudo -n (no interactivo)
# para no bloquear en background.
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  OVS_VSCTL=(ovs-vsctl)
  OVS_OFCTL=(ovs-ofctl)
  TC=(tc)
else
  OVS_VSCTL=(sudo -n ovs-vsctl)
  OVS_OFCTL=(sudo -n ovs-ofctl)
  TC=(sudo -n tc)
  if ! sudo -n true >/dev/null 2>&1; then
    echo "ERROR: ovs/tc requiere sudo, pero no hay sesión sudo válida (sudo -n falla)." >&2
    echo "Solución rápida: ejecuta run_experiment desde una terminal y acepta la contraseña una sola vez (sudo -v)," >&2
    echo "o configura NOPASSWD para ovs-vsctl/ovs-ofctl/tc." >&2
    exit 1
  fi
fi

OUTPUT_DIR="./link_metrics"
INTERVAL=1
QUEUE_ANALYZER=0
LINKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) OUTPUT_DIR="$2"; shift 2;;
    -i|--interval) INTERVAL="$2"; shift 2;;
    --queue-analyzer) QUEUE_ANALYZER=1; shift;;
    -*)
      echo "Unknown option: $1"
      echo "Usage: $0 link1 link2 ... -o <dir> [-i interval] [--queue-analyzer]"
      exit 1
      ;;
    *) LINKS+=("$1"); shift;;
  esac
done

if [ ${#LINKS[@]} -eq 0 ]; then
  echo "Usage: $0 link1 link2 ... -o <dir> [-i interval] [--queue-analyzer]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
}

get_ofport_once() {
  local iface="$1"
  local ofp
  ofp=$("${OVS_VSCTL[@]}" --if-exists get Interface "$iface" ofport 2>/dev/null | tr -d '[:space:]' || true)
  if [ -n "$ofp" ] && [ "$ofp" != "-1" ]; then
    echo "$ofp"
    return 0
  fi
  echo ""
  return 1
}

get_port_counters() {
  local br="$1" ofport="$2"
  "${OVS_OFCTL[@]}" -O OpenFlow13 dump-ports "$br" "$ofport" 2>/dev/null | \
    awk '
      /rx pkts=/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^bytes=/) { rb = $i; gsub("bytes=", "", rb); gsub(",", "", rb) }
          if ($i ~ /^drop=/)  { rd = $i; gsub("drop=",  "", rd); gsub(",", "", rd) }
        }
      }
      /tx pkts=/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^bytes=/) { tb = $i; gsub("bytes=", "", tb); gsub(",", "", tb) }
          if ($i ~ /^drop=/)  { td = $i; gsub("drop=",  "", td); gsub(",", "", td) }
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

get_iface_backlog_bytes() {
  local iface="$1"
  local line val
  line=$("${TC[@]}" -s qdisc show dev "$iface" 2>/dev/null | grep "backlog" | head -n1 || true)
  [ -z "$line" ] && { echo 0; return; }
  val=$(echo "$line" | awk '{print $2}')
  val=${val%b}
  local last=${val: -1}
  local num unit factor
  if [[ "$last" =~ [KkMm] ]]; then
    unit="$last"; num="${val%$last}"
  else
    unit=""; num="$val"
  fi
  case "$unit" in
    K|k) factor=1000;;
    M|m) factor=1000000;;
    *) factor=1;;
  esac
  [ -z "$num" ] && { echo 0; return; }
  awk -v n="$num" -v f="$factor" 'BEGIN { printf "%d\n", n * f }'
}

SWITCHES=()
add_switch_if_new() {
  local sw="$1"
  for s in "${SWITCHES[@]}"; do [ "$s" = "$sw" ] && return; done
  SWITCHES+=("$sw")
}

declare -a LINK_LABEL
declare -a A_SWITCH A_IFACE A_OFPORT
declare -a B_SWITCH B_IFACE B_OFPORT
declare -a LINK_CAP_MBPS
declare -a PREV_TS

idx=0
for link in "${LINKS[@]}"; do
  cap_mbps=0
  if [[ "$link" == *,* ]]; then
    IFS=',' read -r part1 part2 part3 <<< "$link"
    raw_sw1_if1=$(trim "$part1")
    raw_sw2_if2=$(trim "$part2")
    cap_mbps=$(trim "$part3")
    IFS='-' read -r sw1 if1 <<< "$raw_sw1_if1"
    IFS='-' read -r sw2 if2 <<< "$raw_sw2_if2"
  else
    IFS='-' read -r sw1 if1 sw2 if2 <<< "$link"
    cap_mbps=0
  fi

  [ -z "$sw1" ] || [ -z "$if1" ] || [ -z "$sw2" ] || [ -z "$if2" ] && {
    echo "Invalid link format: $link"; exit 1; }

  local_iface1="${sw1}-${if1}"
  local_iface2="${sw2}-${if2}"

  LINK_LABEL[$idx]="${sw1}-${if1}-${sw2}-${if2}"
  LINK_CAP_MBPS[$idx]="$cap_mbps"

  A_SWITCH[$idx]="$sw1"
  A_IFACE[$idx]="$local_iface1"
  A_OFPORT[$idx]=""   # lazy

  B_SWITCH[$idx]="$sw2"
  B_IFACE[$idx]="$local_iface2"
  B_OFPORT[$idx]=""   # lazy

  add_switch_if_new "$sw1"
  add_switch_if_new "$sw2"
  idx=$((idx+1))
done

LINK_COUNT=${#LINK_LABEL[@]}

for ((i=0; i<LINK_COUNT; i++)); do
  csv_file="${OUTPUT_DIR}/${LINK_LABEL[$i]}.csv"
  echo "timestamp,mbps,rx_drop,tx_drop,total_drop,avg_queue_bytes,util_pct" > "$csv_file"
done

declare -a PREV_A_RX_BYTES PREV_A_TX_BYTES PREV_A_RX_DROP PREV_A_TX_DROP
declare -a PREV_B_RX_BYTES PREV_B_TX_BYTES PREV_B_RX_DROP PREV_B_TX_DROP
declare -a HAVE_PREV_A HAVE_PREV_B

for ((i=0; i<LINK_COUNT; i++)); do
  PREV_A_RX_BYTES[$i]=""; PREV_A_TX_BYTES[$i]=""; PREV_A_RX_DROP[$i]=""; PREV_A_TX_DROP[$i]=""
  PREV_B_RX_BYTES[$i]=""; PREV_B_TX_BYTES[$i]=""; PREV_B_RX_DROP[$i]=""; PREV_B_TX_DROP[$i]=""
  HAVE_PREV_A[$i]=0; HAVE_PREV_B[$i]=0; PREV_TS[$i]=""
done

echo "============================================="
echo "  STARTING OVS LINK METRICS CAPTURE (v2)"
echo "  Links:"
for ((i=0; i<LINK_COUNT; i++)); do
  cap="${LINK_CAP_MBPS[$i]}"
  echo "    ${LINK_LABEL[$i]} (cap=${cap} Mbps)"
done
echo "  Interval:   ${INTERVAL}s"
echo "  Folder:     $OUTPUT_DIR"
echo "  Queue analyzer: $QUEUE_ANALYZER"
echo "============================================="

trap 'exit 0' INT TERM
while true; do
  TS=$(date +%s)

  for ((i=0; i<LINK_COUNT; i++)); do
    a_sw="${A_SWITCH[$i]}"; a_iface="${A_IFACE[$i]}"; a_port="${A_OFPORT[$i]}"
    b_sw="${B_SWITCH[$i]}"; b_iface="${B_IFACE[$i]}"; b_port="${B_OFPORT[$i]}"
    cap_mbps="${LINK_CAP_MBPS[$i]}"
    csv_file="${OUTPUT_DIR}/${LINK_LABEL[$i]}.csv"
    prev_ts="${PREV_TS[$i]}"

    # Lazy ofport resolution
    if [ -z "$a_port" ]; then
      a_port=$(get_ofport_once "$a_iface" || true)
      [ -n "$a_port" ] && A_OFPORT[$i]="$a_port"
    fi
    if [ -z "$b_port" ]; then
      b_port=$(get_ofport_once "$b_iface" || true)
      [ -n "$b_port" ] && B_OFPORT[$i]="$b_port"
    fi

    # If any side missing, still log 0 (no blocking)
    if [ -z "$a_port" ] || [ -z "$b_port" ]; then
      backlog_a=$(get_iface_backlog_bytes "$a_iface")
      backlog_b=$(get_iface_backlog_bytes "$b_iface")
      avg_queue_bytes=$(awk -v a="$backlog_a" -v b="$backlog_b" 'BEGIN { print (a + b) / 2.0 }')
      echo "$TS,0,0,0,0,$avg_queue_bytes,0" >> "$csv_file"
      PREV_TS[$i]="$TS"
      continue
    fi

    read a_rx_bytes a_tx_bytes a_rx_drop a_tx_drop < <(get_port_counters "$a_sw" "$a_port")
    read b_rx_bytes b_tx_bytes b_rx_drop b_tx_drop < <(get_port_counters "$b_sw" "$b_port")

    if [ "${HAVE_PREV_A[$i]}" -eq 1 ]; then
      d_a_rx_bytes=$((a_rx_bytes - PREV_A_RX_BYTES[$i])); [ "$d_a_rx_bytes" -lt 0 ] && d_a_rx_bytes=0
      d_a_tx_bytes=$((a_tx_bytes - PREV_A_TX_BYTES[$i])); [ "$d_a_tx_bytes" -lt 0 ] && d_a_tx_bytes=0
      d_a_rx_drop=$((a_rx_drop - PREV_A_RX_DROP[$i]));   [ "$d_a_rx_drop" -lt 0 ] && d_a_rx_drop=0
      d_a_tx_drop=$((a_tx_drop - PREV_A_TX_DROP[$i]));   [ "$d_a_tx_drop" -lt 0 ] && d_a_tx_drop=0
    else
      d_a_rx_bytes=0; d_a_tx_bytes=0; d_a_rx_drop=0; d_a_tx_drop=0
    fi
    if [ "${HAVE_PREV_B[$i]}" -eq 1 ]; then
      d_b_rx_bytes=$((b_rx_bytes - PREV_B_RX_BYTES[$i])); [ "$d_b_rx_bytes" -lt 0 ] && d_b_rx_bytes=0
      d_b_tx_bytes=$((b_tx_bytes - PREV_B_TX_BYTES[$i])); [ "$d_b_tx_bytes" -lt 0 ] && d_b_tx_bytes=0
      d_b_rx_drop=$((b_rx_drop - PREV_B_RX_DROP[$i]));   [ "$d_b_rx_drop" -lt 0 ] && d_b_rx_drop=0
      d_b_tx_drop=$((b_tx_drop - PREV_B_TX_DROP[$i]));   [ "$d_b_tx_drop" -lt 0 ] && d_b_tx_drop=0
    else
      d_b_rx_bytes=0; d_b_tx_bytes=0; d_b_rx_drop=0; d_b_tx_drop=0
    fi

    total_rx_drop=$((d_a_rx_drop + d_b_rx_drop))
    total_tx_drop=$((d_a_tx_drop + d_b_tx_drop))
    total_drop=$((total_rx_drop + total_tx_drop))

    total_a_bytes=$((d_a_rx_bytes + d_a_tx_bytes))
    total_b_bytes=$((d_b_rx_bytes + d_b_tx_bytes))

    dt=$INTERVAL
    if [ -n "$prev_ts" ]; then
      dt=$((TS - prev_ts)); [ "$dt" -le 0 ] && dt=$INTERVAL
    fi

    mbps=$(LC_NUMERIC=C awk -v a="$total_a_bytes" -v b="$total_b_bytes" -v dt="$dt" '
      BEGIN {
        if (dt <= 0) { printf "0.000\n"; exit }
        total_bytes = (a + b) / 2.0
        mbps = (total_bytes * 8.0) / (dt * 1000000.0)
        printf "%.3f\n", mbps
      }')

    backlog_a=$(get_iface_backlog_bytes "$a_iface")
    backlog_b=$(get_iface_backlog_bytes "$b_iface")
    avg_queue_bytes=$(awk -v a="$backlog_a" -v b="$backlog_b" 'BEGIN { print (a + b) / 2.0 }')

    util_pct=$(LC_NUMERIC=C awk -v mb="$mbps" -v cap="$cap_mbps" '
      BEGIN {
        if (cap + 0 <= 0) { printf "0.000\n"; exit }
        printf "%.3f\n", (mb / cap) * 100.0
      }')

    echo "$TS,$mbps,$total_rx_drop,$total_tx_drop,$total_drop,$avg_queue_bytes,$util_pct" >> "$csv_file"

    PREV_A_RX_BYTES[$i]=$a_rx_bytes; PREV_A_TX_BYTES[$i]=$a_tx_bytes
    PREV_A_RX_DROP[$i]=$a_rx_drop;   PREV_A_TX_DROP[$i]=$a_tx_drop
    PREV_B_RX_BYTES[$i]=$b_rx_bytes; PREV_B_TX_BYTES[$i]=$b_tx_bytes
    PREV_B_RX_DROP[$i]=$b_rx_drop;   PREV_B_TX_DROP[$i]=$b_tx_drop
    HAVE_PREV_A[$i]=1; HAVE_PREV_B[$i]=1
    PREV_TS[$i]="$TS"
  done

  if [ "$QUEUE_ANALYZER" -eq 1 ]; then
    for sw in "${SWITCHES[@]}"; do
      "${OVS_OFCTL[@]}" -O OpenFlow13 dump-queue-stats "$sw" 2>/dev/null | \
        awk -v ts="$TS" -v sw="$sw" -v outdir="$OUTPUT_DIR" '
          /port [0-9]+: queue_id [0-9]+:/ {
            port = ""; qid = "";
            for (i = 1; i <= NF; i++) {
              if ($i ~ /^port$/) { port = $(i+1); sub(":", "", port); }
              if ($i ~ /^queue_id$/) { qid = $(i+1); sub(":", "", qid); }
            }
            getline;
            txb = txp = txe = 0;
            for (i = 1; i <= NF; i++) {
              if ($i ~ /^tx_bytes=/) { txb = $i; gsub("tx_bytes=", "", txb); gsub(",", "", txb); }
              if ($i ~ /^tx_packets=/) { txp = $i; gsub("tx_packets=", "", txp); gsub(",", "", txp); }
              if ($i ~ /^tx_errors=/) { txe = $i; gsub("tx_errors=", "", txe); gsub(",", "", txe); }
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