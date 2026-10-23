#!/bin/bash
# 다음 실험 대기열: (1) 팀 전달용 TRP 모델 -> (2) MACs 예산 TRP (시험 실행 후 본 실행, 프로세스 3개) -> 요약
# 사용: nohup bash run_next.sh > work/next_queue.log 2>&1 &
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
while pgrep -f "trp_search.py|gradual_prune.py|prune_grid.py|stage_sensitivity.py" > /dev/null; do sleep 60; done

log "(1) 팀 전달용 모델: 팀 분할 seed0, trp_bn, 파라미터 16% / 4%"
python make_team_trp.py > work/team_trp.log 2>&1
log "(1) 끝: $(ls work/team_trp/*.pt 2>/dev/null | wc -l)개 모델"

log "(2) MACs 예산 TRP 시험 실행"
python trp_search.py --tasks pair_gl-pl --folds 0 --resource macs --budgets 0.372 --methods trp_bn \
  --out-name smoke_trp_macs --workers 4 > work/smoke_trp_macs.log 2>&1
if [ ! -f work/runs/smoke_trp_macs/pair_gl-pl/fold0_seed0/trp_bn_b0.372.json ]; then
  log "(2) 시험 실행 실패. work/smoke_trp_macs.log를 확인하세요."
  exit 1
fi

log "(2) MACs 예산 TRP 본 실행: 5과제 x 5 fold x 방법 3개 x 예산 4개"
shards trp_macs python trp_search.py --resource macs --budgets 0.372 0.173 0.048 0.015 \
  --methods trp_bn trp global_l2 --out-name trp_macs
python summarize_trp.py --out-name trp_macs --resource macs > work/trp_macs_summary.log 2>&1
log "전부 끝"
