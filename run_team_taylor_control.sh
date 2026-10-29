#!/bin/bash
# Shared-probe C/E ablation. Does not alter old A/B/C results or baseline models.
# Usage: nohup bash run_team_taylor_control.sh > work/team_taylor_control_queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p work
exec 9> work/team_taylor_control.lock
if ! flock -n 9; then
  echo 'Taylor control queue already running.'
  exit 1
fi
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log 'Checking existing inputs and CUDA (no baseline training).'
python team_taylor_control.py check --device cuda
log 'Waiting for existing experiments to finish.'
python wait_idle.py
log 'Smoke: seed6, both matched arms, 16 recovery steps.'
# Unique directory: never delete or reuse an older smoke result.
smoke="team_taylor_control_smoke_$(date '+%Y%m%d_%H%M%S')_$$"
python team_taylor_control.py train --device cuda --seeds 6 --steps 16 --out-name "$smoke" \
  > work/team_taylor_control_smoke.log 2>&1
python team_taylor_control.py judge --device cuda --seeds 6 --steps 16 --out-name "$smoke" \
  >> work/team_taylor_control_smoke.log 2>&1
log 'Full run: seed6-10, ratio0.4, C/E, 2800 recovery steps each (10 models).'
python team_taylor_control.py train --device cuda > work/team_taylor_control_train.log 2>&1
python team_taylor_control.py judge --device cuda > work/team_taylor_control_result.txt 2>&1
log 'Done: work/team_taylor_control_result.txt'
