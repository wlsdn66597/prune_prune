#!/bin/bash
# 지금 돌고 있는 곡선 실험(prune_grid.py --recover-steps 840)이 끝나면 아래 실험을 순서대로 돌린다.
# 사용: nohup bash run_queue.sh > work/queue.log 2>&1 &
cd "$(dirname "$0")"
source .venv/bin/activate

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

# 3개 프로세스로 나눠 돌리고 모두 끝날 때까지 기다린다. 사용: shards <로그 이름> <명령...>
shards() {
  local name=$1; shift
  for i in 0 1 2; do
    "$@" --shard $i --nshards 3 --workers 4 > "work/${name}_$i.log" 2>&1 &
  done
  wait
}

log "곡선 실험이 끝나기를 기다리는 중"
while pgrep -f "prune_grid.py --recover-steps 840" > /dev/null; do sleep 60; done
log "곡선 실험 끝, 결과 요약"
python summarize_prune.py --recover-steps 840 > work/rec840_summary.log 2>&1

log "3번: 56 step 회복 시작"
shards rec56 python prune_grid.py --recover-steps 56 --ratios 0.6 0.8
python select_by_recovery.py > work/select.log 2>&1
log "3번 끝 ($(find work/runs/prune -name '*_rec56.json' | wc -l)/1320)"

log "4번: 6단계 시험 실행"
python stage_sensitivity.py --tasks pair_gl-pl --folds 0 --criteria l2 --stages layer4 --ratios 0.5 \
  --out-name smoke_stage --workers 4 > work/smoke_stage.log 2>&1
ok=$(python -c "import json,glob; r=json.load(open(glob.glob('work/runs/smoke_stage/pair_gl-pl/fold0_seed0/*.json')[0])); print(int(r['params_ratio'] < 0.99))" 2>/dev/null)
if [ "$ok" != "1" ]; then
  log "6단계 시험 실행에서 아무것도 잘리지 않았거나 에러가 났습니다. work/smoke_stage.log 확인 후 수동으로 돌리세요."
  exit 1
fi

log "4번: 6단계 자른 직후"
shards stage0 python stage_sensitivity.py
python stage_sensitivity.py --summary > work/stage0_summary.log 2>&1

log "4번: 6단계 회복 후"
shards stage840 python stage_sensitivity.py --recover-steps 840
python stage_sensitivity.py --summary --recover-steps 840 > work/stage840_summary.log 2>&1

log "전부 끝"
