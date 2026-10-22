#!/bin/bash
# TRP 실험 대기열: 예산 예측 검증 -> 시험 실행 -> 본 실행(프로세스 3개) -> 요약
# 사용: nohup bash run_trp.sh > work/trp_queue.log 2>&1 &
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
while pgrep -f "prune_grid.py|stage_sensitivity.py|train_base.py" > /dev/null; do sleep 60; done

log "0단계 검증: 유사도로 프루닝 여유 예측 (학습 없음)"
python predict_budget.py > work/predict_budget.log 2>&1

log "시험 실행 (pair_gl-pl fold0, 예산 0.36, trp와 global_l2)"
python trp_search.py --tasks pair_gl-pl --folds 0 --budgets 0.36 --methods trp global_l2 \
  --out-name smoke_trp --workers 4 > work/smoke_trp.log 2>&1
n=$(ls work/runs/smoke_trp/pair_gl-pl/fold0_seed0/*_b0.36.json 2>/dev/null | wc -l)
if [ "$n" != "2" ]; then
  log "시험 실행 실패. work/smoke_trp.log를 확인하세요."
  exit 1
fi

log "본 실행: 5과제 x 5 fold x 방법 5개 x 예산 4개"
shards trp python trp_search.py
python summarize_trp.py > work/trp_summary.log 2>&1
log "전부 끝"
