#!/usr/bin/env bash
# ========================================
# run_hackrf.sh
# Runs two HackRF transfers in parallel for a fixed duration.
# Usage:
#   ./run_hackrf.sh <seconds> [gui]
#
#   <seconds> : duration to run both transfers
#   [gui]     : optional, opens two macOS Terminal windows (macOS only)
# ========================================

set -euo pipefail

DURATION="${1:-}"
MODE="${2:-}"
if [[ -z "$DURATION" ]]; then
  echo "Usage: $0 DURATION_SECONDS [gui]"
  exit 1
fi

# --- HackRF commands ---
CMD1='hackrf_transfer -H -d 000000000000000066a062dc226b169f -a 0 -l 24 -g 50 -r 63.cs8 -f 500000000 -s 20000000'
CMD2='hackrf_transfer -d 0000000000000000436c63dc38284c63 -a 0 -l 24 -g 50 -r 9f.cs8 -f 500000000 -s 20000000'

LOG1="63.log"
LOG2="9f.log"
OUT1="63.cs8"
OUT2="9f.cs8"

# Clean up any old files
rm -f "$LOG1" "$LOG2" "$OUT1" "$OUT2"

# Track PIDs for cleanup
pids_to_kill=()
cleanup() {
  if [[ ${#pids_to_kill[@]} -gt 0 ]]; then
    echo
    echo "Cleaning up: terminating processes ${pids_to_kill[*]} ..."
    kill "${pids_to_kill[@]}" 2>/dev/null || true
    sleep 1
    for p in "${pids_to_kill[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        echo "Process $p still alive; sending SIGKILL..."
        kill -9 "$p" 2>/dev/null || true
      fi
    done
  fi
}
trap cleanup EXIT INT TERM

# Platform detection
is_macos=false
case "$OSTYPE" in
  darwin*) is_macos=true ;;
esac

# --------------------------
# GUI mode (macOS Terminal)
# --------------------------
if [[ "$MODE" == "gui" && "$is_macos" == true ]]; then
  echo "Opening two new Terminal windows and launching commands (macOS)..."

  TERM_CMD1="rm -f $LOG1 $OUT1; $CMD1 > $LOG1 2>&1"
  TERM_CMD2="rm -f $LOG2 $OUT2; $CMD2 > $LOG2 2>&1"

  osascript <<EOF
tell application "Terminal"
  activate
  do script "$TERM_CMD1"
  do script "$TERM_CMD2"
end tell
EOF

  echo "Waiting 1s for initial output..."
  sleep 1
  echo
  echo "=== Initial 5 lines from $LOG1 ==="
  head -n 5 "$LOG1" 2>/dev/null || echo "(no output yet)"
  echo
  echo "=== Initial 5 lines from $LOG2 ==="
  head -n 5 "$LOG2" 2>/dev/null || echo "(no output yet)"

  echo
  echo "Running for $DURATION seconds..."
  sleep "$DURATION"

  echo
  echo "Time's up: stopping both HackRF transfers..."
  pkill -f '0000000000000000436c63dc38284c63' || true
  pkill -f '000000000000000066a062dc226b169f' || true
  pkill -f hackrf_transfer || true

  sleep 1
  echo
  echo "=== Final 5 lines from $LOG1 ==="
  tail -n 5 "$LOG1" 2>/dev/null || echo "(no log found)"
  echo
  echo "=== Final 5 lines from $LOG2 ==="
  tail -n 5 "$LOG2" 2>/dev/null || echo "(no log found)"

  # --- Crop both files to same sample count ---
  echo
  echo "Cropping both .cs8 files to the same length..."

  if stat --version >/dev/null 2>&1; then
    SIZE1=$(stat -c%s "$OUT1")
    SIZE2=$(stat -c%s "$OUT2")
  else
    SIZE1=$(stat -f%z "$OUT1")
    SIZE2=$(stat -f%z "$OUT2")
  fi

  if (( SIZE1 < SIZE2 )); then
    MIN_SIZE=$SIZE1
  else
    MIN_SIZE=$SIZE2
  fi
  (( MIN_SIZE %= 2 == 0 )) || ((MIN_SIZE--))

  echo "  $OUT1: $SIZE1 bytes"
  echo "  $OUT2: $SIZE2 bytes"
  echo "  Cropping both to $MIN_SIZE bytes (=$((MIN_SIZE/2)) samples)."

  dd if="$OUT1" of="${OUT1}.tmp" bs=1 count="$MIN_SIZE" status=none
  mv "${OUT1}.tmp" "$OUT1"
  dd if="$OUT2" of="${OUT2}.tmp" bs=1 count="$MIN_SIZE" status=none
  mv "${OUT2}.tmp" "$OUT2"

  echo "Cropping complete. Both files now equal in length."
  echo
  echo "Done. Files: $OUT1, $OUT2 — Logs: $LOG1, $LOG2"
  exit 0
fi

# --------------------------
# Default: background mode
# --------------------------
echo "Starting both commands in background..."
nohup bash -lc "$CMD1" > "$LOG1" 2>&1 &
PID1=$!
pids_to_kill+=("$PID1")

nohup bash -lc "$CMD2" > "$LOG2" 2>&1 &
PID2=$!
pids_to_kill+=("$PID2")

echo "Started PID1=$PID1 PID2=$PID2"
sleep 1

echo
echo "=== Initial 5 lines from $LOG1 ==="
head -n 5 "$LOG1" 2>/dev/null || echo "(no output yet)"
echo
echo "=== Initial 5 lines from $LOG2 ==="
head -n 5 "$LOG2" 2>/dev/null || echo "(no output yet)"

echo
echo "Running for $DURATION seconds..."
sleep "$DURATION"

echo
echo "Time's up! Terminating both processes..."
kill "$PID1" "$PID2" 2>/dev/null || true
sleep 1
for p in "$PID1" "$PID2"; do
  if kill -0 "$p" 2>/dev/null; then
    echo "Process $p still alive; forcing..."
    kill -9 "$p" 2>/dev/null || true
  fi
done
pids_to_kill=()

echo
echo "=== Final 5 lines from $LOG1 ==="
tail -n 5 "$LOG1" 2>/dev/null || echo "(no log found)"
echo
echo "=== Final 5 lines from $LOG2 ==="
tail -n 5 "$LOG2" 2>/dev/null || echo "(no log found)"

if stat --version >/dev/null 2>&1; then
  SIZE1=$(stat -c%s "$OUT1")
  SIZE2=$(stat -c%s "$OUT2")
else
  SIZE1=$(stat -f%z "$OUT1")
  SIZE2=$(stat -f%z "$OUT2")
fi
echo
echo "  $OUT1: $SIZE1 bytes"
echo "  $OUT2: $SIZE2 bytes"
echo
echo "Done. Files: $OUT1, $OUT2 — Logs: $LOG1, $LOG2"






















































































































































































































