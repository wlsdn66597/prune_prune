#!/bin/bash
# 파인튜닝 생략·가중치 상속 실험 대기열: 대기 -> 시험 실행 -> 사전학습 L2(840, 2800 step)
#   -> 사전학습 가중치 + movement 마스크(2800) -> 기성 소형 모델 -> 지연시간 다시 측정 -> 요약
# 22과제 x fold 0·1 x 비율 {0.8, 0.9}. 이미 끝난 조합은 건너뛴다. 예상 약 2.5시간.
# 사용: nohup bash run_inherit.sh > work/inherit_queue.log 2>&1 &
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

exec 9> work/inherit.lock
if ! flock -n 9; then
  log "run_inherit.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행 (사전학습 가중치 자르기 16 step, MobileNet 2종 20 step)"
rm -rf work/runs/smoke_inherit work/runs/smoke_mobile2
python prune_grid.py --init pretrained --tasks pair_gl-pl --folds 0 --criteria l2 mov_cum --ratios 0.9 \
  --recover-steps 16 --out-name smoke_inherit --workers 4 > work/smoke_inherit.log 2>&1
python train_mobile.py --tasks pair_gl-pl --folds 0 --archs mnv3s mnv2 --lrs 1e-3 --total-steps 20 --eval-every 10 \
  --out-name smoke_mobile2 --workers 4 >> work/smoke_inherit.log 2>&1
n=$(find work/runs/smoke_inherit work/runs/smoke_mobile2 -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 4 ]; then
  log "시험 실행 실패 (결과 $n/4개). work/smoke_inherit.log를 확인하세요."
  exit 1
fi

GRID="python prune_grid.py --init pretrained --folds 0 1 --ratios 0.8 0.9"
log "사전학습 가중치를 L2로 자르고 840 step 학습"
shards inherit840 $GRID --criteria l2 --recover-steps 840
log "사전학습 가중치 (L2 마스크, movement 마스크) 2800 step 학습"
shards inherit2800 $GRID --criteria l2 mov_cum --recover-steps 2800
log "기성 소형 모델 (MobileNetV3-Small, MobileNetV2, 학습률 2개 중 검증으로 선택)"
shards mobile python train_mobile.py --folds 0 1 --archs mnv3s mnv2
log "지연시간 다시 측정 (서버 CPU)"
python measure_latency.py --threads 1 4 > work/latency_server.txt 2>&1
python summarize_inherit.py > work/inherit_summary.txt 2>&1
log "전부 끝. 결과: work/inherit_summary.txt"
