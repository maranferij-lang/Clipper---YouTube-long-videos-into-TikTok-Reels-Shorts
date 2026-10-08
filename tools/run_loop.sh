#!/bin/bash
# run_loop.sh <jobs.json> [work_dir]
# Native run (Terminal): loops the pipeline until ALL DONE, no time limit. Run setup_mac_native.sh first.
JOBS=${1:?usage: run_loop.sh jobs.json [work_dir]}; WORK=${2:-$(dirname "$JOBS")/w}
source "$HOME/clipper_venv/bin/activate" 2>/dev/null
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$HOME/Light-ASD"
T="$(cd "$(dirname "$0")" && pwd)"
while true; do
  OUT=$(python3 "$T/pipeline_v3.py" "$JOBS" --work "$WORK" --budget 100000 2>&1)
  echo "$OUT" | grep -E "FINAL|STOP|cut ok|!!!|FAIL|стан:|QC"
  echo "$OUT" | grep -q "ALL DONE" && { echo "ALL DONE"; break; }
  echo "$OUT" | grep -q "STOP" || sleep 2
done
