#!/bin/bash
# 과제별 계획 확인 대기열 (파인튜닝 실행만): 대기 -> 시험 실행 -> fold 2·3·4의 파인튜닝 + movement 모델
#   (비율 0.4·0.6·0.7·0.8·0.9, 2800 step 회복) -> 확인 판정
# 22과제 x fold 2·3·4 x 비율 5개 = 330개. 기준 모델과 840 step 결과는 이미 있는 것을 쓴다. 예상 약 3시간.
# 확인 항목(plan_diag.py에 미리 적어 둠): [A] 요구 90%의 과제별 계획 이득, [D] 회복 이득과 난이도.
# 사용: nohup bash run_confirm.sh > work/confirm_queue.log 2>&1 &
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

exec 9> work/confirm.lock
if ! flock -n 9; then
  log "run_confirm.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행 1: 탐색 단계 진단을 다시 돌려 코드 변경 확인 (fold 0·1, GPU 안 씀)"
if ! python plan_diag.py > work/plan_diag_recheck.txt 2>&1; then
  log "진단 시험 실패. work/plan_diag_recheck.txt를 확인하세요."
  exit 1
fi
log "시험 실행 2: movement 자르기 (fold 2, 비율 0.4, 16 step)"
rm -rf work/runs/smoke_confirm
python prune_grid.py --tasks pair_gl-pl --folds 2 --criteria mov_cum --ratios 0.4 \
  --recover-steps 16 --out-name smoke_confirm --workers 4 > work/smoke_confirm.log 2>&1
n=$(find work/runs/smoke_confirm -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 1 ]; then
  log "시험 실행 실패 (결과 $n/1개). work/smoke_confirm.log를 확인하세요."
  exit 1
fi

log "본 실행: fold 2·3·4, 비율 5개, 2800 step (프로세스 3개)"
shards confirm python prune_grid.py --folds 2 3 4 --criteria mov_cum --ratios 0.4 0.6 0.7 0.8 0.9 --recover-steps 2800
log "확인 판정"
python plan_diag.py --folds 2 3 4 > work/plan_confirm.txt 2>&1
log "전부 끝. 결과: work/plan_confirm.txt"
