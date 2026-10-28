#!/bin/bash
# 프루닝·양자화 예산 분할 실험 대기열: 대기 -> 시험 실행 -> 비율 0.4·0.6·0.7 movement 모델(2800 step 회복)
#   -> 크기·정확도 평가 -> 요약
# 22과제 x fold 0·1. 비율 0(원본)·0.8·0.9 모델은 이미 있는 것을 쓴다. 예상: 학습 약 1.2시간 + 평가 약 30분.
# 사용: nohup bash run_split.sh > work/split_queue.log 2>&1 &
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

exec 9> work/split.lock
if ! flock -n 9; then
  log "run_split.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행 1: 평가·분석 스크립트 (이미 있는 원본·0.8 모델, 과제 5개 fold0)"
if ! python split_eval.py --tasks pair_gl-pl,pair_cb-gl,tri_gl-me-pl,tri_cb-gl-me,all6 --folds 0 --ratios 0 0.8 \
    > work/split_smoke_eval.txt 2>&1; then
  log "평가 시험 실패. work/split_smoke_eval.txt를 확인하세요."
  exit 1
fi
log "시험 실행 2: 학습 (비율 0.4, 16 step)"
rm -rf work/runs/qready_smoke
python qready_train.py --smoke --arms 0.4:movement > work/split_smoke_train.log 2>&1
n=$(find work/runs/qready_smoke -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 1 ]; then
  log "학습 시험 실패 (결과 $n/1개). work/split_smoke_train.log를 확인하세요."
  exit 1
fi

log "본 실행: 비율 0.4·0.6·0.7 movement 모델 (프로세스 3개)"
shards split python qready_train.py --arms 0.4:movement 0.6:movement 0.7:movement
log "크기·정확도 평가 (약 30분)"
python split_eval.py > work/split_summary.txt 2>&1
log "전부 끝. 결과: work/split_summary.txt"
