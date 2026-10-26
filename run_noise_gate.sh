#!/bin/bash
# 잡음 확인 실험(판단 관문) 대기열: 다른 학습이 끝나기를 기다림 -> 시험 실행 -> 본 실행(프로세스 3개) -> 요약
# 사용: nohup bash run_noise_gate.sh > work/noise_gate_queue.log 2>&1 &
# 기다리지 않고 바로 시작하려면: pkill -f 'noise_gate.py --wait-only'
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p work

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

exec 9> work/noise_gate.lock
if ! flock -n 9; then
  log "run_noise_gate.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python noise_gate.py --wait-only

log "시험 실행 (모든 조건의 입력과 후보 해시 확인 + 16 step 학습)"
if ! python noise_gate.py --smoke > work/noise_gate_smoke.log 2>&1; then
  log "시험 실행 실패. work/noise_gate_smoke.log를 확인하세요."
  exit 1
fi
tail -n 3 work/noise_gate_smoke.log

log "본 실행: 프로세스 3개"
for i in 0 1 2; do
  python noise_gate.py --shard $i --nshards 3 > "work/noise_gate_$i.log" 2>&1 &
done
wait
python summarize_noise_gate.py > work/noise_gate_summary.txt 2>&1
log "전부 끝. 결과: work/noise_gate_summary.txt"
