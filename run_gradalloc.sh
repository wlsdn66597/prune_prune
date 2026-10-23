#!/bin/bash
# 배분 + 점진 결합 실험 대기열: TRP-J(run_joint.sh)가 끝나면 시험 실행 -> 설정 5개 차례로 (각각 프로세스 3개) -> 요약
# 사용: nohup bash run_gradalloc.sh > work/gradalloc_queue.log 2>&1 &
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

log "앞선 대기열이 끝나기를 기다리는 중"
while pgrep -f "run_joint.sh|trp_joint.py|run_next.sh|trp_search.py|make_team_trp.py|gradual_prune.py|prune_grid.py" > /dev/null; do
  sleep 60
done

log "시험 실행 (pair_gl-pl fold0, TRP 배분, 비율 0.8)"
python gradual_alloc.py --target trp --tasks pair_gl-pl --folds 0 --ratios 0.8 \
  --out-name smoke_gradalloc --workers 4 > work/smoke_gradalloc.log 2>&1
if [ ! -f work/runs/smoke_gradalloc/pair_gl-pl/fold0_seed0/grad_trp_online_e30_r0.8.json ]; then
  log "시험 실행 실패. work/smoke_gradalloc.log를 확인하세요."
  exit 1
fi

log "1/5 TRP 배분, 갱신, 30% 도달"
shards ga1 python gradual_alloc.py --target trp --end-frac 0.3
log "2/5 TRP 배분, 고정, 30% 도달"
shards ga2 python gradual_alloc.py --target trp --end-frac 0.3 --fixed
log "3/5 균일, 갱신, 30% 도달"
shards ga3 python gradual_alloc.py --target uniform --end-frac 0.3
log "4/5 TRP-J 배분, 갱신, 30% 도달"
shards ga4 python gradual_alloc.py --target joint --end-frac 0.3
log "5/5 TRP 배분, 갱신, 60% 도달"
shards ga5 python gradual_alloc.py --target trp --end-frac 0.6

python summarize_gradalloc.py > work/gradalloc_summary.log 2>&1
log "전부 끝"
