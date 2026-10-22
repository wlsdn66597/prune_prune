#!/bin/bash
# GTP 실험 대기열: TRP 대기열이 끝나면 시험 실행 -> 본 실행(프로세스 3개) -> 요약
# 사용: nohup bash run_gradual.sh > work/gradual_queue.log 2>&1 &
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

log "TRP 실험이 끝나기를 기다리는 중"
while pgrep -f "run_trp.sh|trp_search.py|prune_grid.py|stage_sensitivity.py" > /dev/null; do sleep 60; done

log "시험 실행 (pair_gl-pl fold0, gtp, 비율 0.8)"
python gradual_prune.py --tasks pair_gl-pl --folds 0 --ratios 0.8 --methods gtp \
  --out-name smoke_gradual --workers 4 > work/smoke_gradual.log 2>&1
ok=$(python -c "import json; r=json.load(open('work/runs/smoke_gradual/pair_gl-pl/fold0_seed0/gtp_r0.8.json')); print(int(r['params_ratio'] < 0.1))" 2>/dev/null)
if [ "$ok" != "1" ]; then
  log "시험 실행 실패. work/smoke_gradual.log를 확인하세요."
  exit 1
fi

log "본 실행: 22과제 x 5 fold x 방법 3개 x 비율 3개"
shards gradual python gradual_prune.py
python summarize_gradual.py > work/gradual_summary.log 2>&1
log "전부 끝"
