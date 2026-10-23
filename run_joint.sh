#!/bin/bash
# TRP-J 대기열: 시험 실행 -> 본 실행(프로세스 3개) -> 요약
# 사용: nohup bash run_joint.sh > work/joint_queue.log 2>&1 &
cd "$(dirname "$0")"
source .venv/bin/activate

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

shards() {
  local name=$1; shift
  for i in 0 1 2; do
    "$@" --shard $i --nshards 3 --workers 4 > "work/${name}_$i.log" 2>&1 &
  done
  wait
}

log "다른 실험이 끝나기를 기다리는 중"
while pgrep -f "run_next.sh|trp_search.py|make_team_trp.py|gradual_prune.py|prune_grid.py" > /dev/null; do sleep 60; done

log "시험 실행 (pair_gl-pl fold0, 비율 0.8, 재배분 1회 x 후보 4개)"
python trp_joint.py --tasks pair_gl-pl --folds 0 --ratios 0.8 --rounds 1 --moves 4 \
  --out-name smoke_joint --workers 4 > work/smoke_joint.log 2>&1
if [ ! -f work/runs/smoke_joint/pair_gl-pl/fold0_seed0/trp_joint_r0.8.json ]; then
  log "시험 실행 실패. work/smoke_joint.log를 확인하세요."
  exit 1
fi

log "본 실행: 5과제 x 5 fold x 균일 비율 4개 (0.4 / 0.6 / 0.8 / 0.9)"
shards joint python trp_joint.py
python summarize_joint.py > work/joint_summary.log 2>&1
log "전부 끝"
