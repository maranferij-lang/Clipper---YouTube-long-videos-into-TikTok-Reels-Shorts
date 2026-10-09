#!/bin/bash
# run_ep.sh <jobs.json> [budget_seconds] [work_dir]
# One step of the pipeline. In a sandbox with a per-call time limit, call it repeatedly until it prints ALL DONE.
JOBS=${1:?usage: run_ep.sh jobs.json [budget] [work_dir]}; BUD=${2:-165}; WORK=${3:-$(dirname "$JOBS")/w}
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$HOME/Light-ASD"
T="$(cd "$(dirname "$0")" && pwd)"
python3 "$T/pipeline_v3.py" "$JOBS" --work "$WORK" --budget "$BUD" 2>&1 | grep -E "FINAL|ALL DONE|STOP|cut ok|asd |hq |!!!|FAIL|state:|QC" | tail -14
df -h "$HOME" | tail -1 | awk '{print "[disk] free", $4}'
