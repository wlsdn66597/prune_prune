#!/bin/bash
# 미리 정한 프루닝 계획 검증 대기열: 미리 보기 요약 -> 대기 -> 시험 실행
#   -> 파인튜닝 생략 모델(사전학습 가중치를 L2로 자르고 2800 step 학습) 비율 0.4·0.6·0.7 -> 요약
# 22과제 x fold 0·1. 0.8·0.9 생략 모델(상속 실험)과 파인튜닝 + movement 모델(예산 분할 실험)은 이미 있는 것을 쓴다.
# 예상: 학습 약 1.2~1.5시간 + 요약 1분.
# 사용: nohup bash run_plan.sh > work/plan_queue.log 2>&1 &
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

exec 9> work/plan.lock
if ! flock -n 9; then
  log "run_plan.sh가 이미 실행 중입니다."
  exit 1
fi

log "미리 보기: 이미 있는 결과로 계획 검증 요약 (GPU 안 씀)"
if ! python plan_eval.py > work/plan_preview.txt 2>&1; then
  log "요약 실패. work/plan_preview.txt를 확인하세요."
  exit 1
fi
log "미리 보기 결과: work/plan_preview.txt"

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행: 사전학습 가중치 L2 자르기 (비율 0.4, 16 step)"
rm -rf work/runs/smoke_plan
python prune_grid.py --init pretrained --tasks pair_gl-pl --folds 0 --criteria l2 --ratios 0.4 \
  --recover-steps 16 --out-name smoke_plan --workers 4 > work/smoke_plan.log 2>&1
n=$(find work/runs/smoke_plan -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 1 ]; then
  log "시험 실행 실패 (결과 $n/1개). work/smoke_plan.log를 확인하세요."
  exit 1
fi

log "본 실행: 파인튜닝 생략 모델 비율 0.4·0.6·0.7 (프로세스 3개)"
shards plan python prune_grid.py --init pretrained --folds 0 1 --criteria l2 --ratios 0.4 0.6 0.7 --recover-steps 2800
log "계획 검증 요약"
python plan_eval.py > work/plan_summary.txt 2>&1
log "전부 끝. 결과: work/plan_summary.txt"
