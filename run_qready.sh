#!/bin/bash
# 양자화 대비 프루닝 1단계 대기열: 대기 -> 시험 실행 -> 본 실행(프로세스 3개) -> 요약
# 22과제 x fold 0·1, 비율 0.8(movement, l2, qr_full@1.0, qr_full@4.0)과 0.9(movement, qr_full@1.0), 2800 step 회복.
# 예상 약 2.5시간. 자른 모델은 work/runs/qready/에 .pt로 저장된다 (팀원 양자화 비교용).
# 사용: nohup bash run_qready.sh > work/qready_queue.log 2>&1 &
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

exec 9> work/qready.lock
if ! flock -n 9; then
  log "run_qready.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행 (pair_gl-pl fold0, 비율 0.8, movement·qr_full@1.0, 16 step)"
rm -rf work/runs/qready_smoke
python qready_train.py --smoke > work/qready_smoke.log 2>&1
n=$(find work/runs/qready_smoke -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 2 ]; then
  log "시험 실행 실패 (결과 $n/2개). work/qready_smoke.log를 확인하세요."
  exit 1
fi

log "본 실행: 프로세스 3개"
shards qready python qready_train.py
python qready_train.py --summary > work/qready_summary.txt 2>&1
log "전부 끝. 결과: work/qready_summary.txt"
