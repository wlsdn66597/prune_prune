#!/bin/bash
# 기준 x 회복 예산 실험 대기열: 대기 -> 시험 실행 -> 840 step 빈 칸 채우기 -> 2800 step -> 요약
# 22과제 x fold 0·1 x 기준 {mov_cum, l2, random} x 비율 {0.8, 0.9}. 이미 끝난 조합은 prune_grid가 건너뛴다.
# 사용: nohup bash run_budget.sh > work/budget_queue.log 2>&1 &
# 기다리지 않고 바로 시작하려면: pkill -f wait_idle.py
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p work

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

shards() {
  local name=$1; shift
  for i in 0 1 2; do
    "$@" --shard $i --nshards 3 --workers 4 > "work/${name}_$i.log" 2>&1 &
  done
  wait
}

exec 9> work/budget.lock
if ! flock -n 9; then
  log "run_budget.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행 (pair_gl-pl fold0, random 0.9, 16 step)"
rm -rf work/runs/smoke_budget
python prune_grid.py --tasks pair_gl-pl --folds 0 --criteria random --ratios 0.9 --recover-steps 16 \
  --out-name smoke_budget --workers 4 > work/smoke_budget.log 2>&1
if [ ! -f work/runs/smoke_budget/pair_gl-pl/fold0_seed0/random_r0.9_rec16.json ]; then
  log "시험 실행 실패. work/smoke_budget.log를 확인하세요."
  exit 1
fi
# 요약 스크립트 시험: 이미 있는 56·840 step 결과로 돌려 본다 (실패해도 실험은 계속)
if ! python summarize_budget.py --budgets 56 840 --ratios 0.8 > work/budget_preview_56_840.txt 2>&1; then
  log "요약 스크립트 시험 실패 (실험은 계속). work/budget_preview_56_840.txt 확인"
fi

GRID="python prune_grid.py --folds 0 1 --criteria mov_cum l2 random --ratios 0.8 0.9"
log "840 step 빈 칸 채우기"
shards budget840 $GRID --recover-steps 840
log "2800 step (약 3.5~4시간)"
shards budget2800 $GRID --recover-steps 2800
python summarize_budget.py > work/budget_summary.txt 2>&1
log "전부 끝. 결과: work/budget_summary.txt"
