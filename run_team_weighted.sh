#!/bin/bash
# 분리도 가중 회복 대기열: 대기 -> 시험 실행 -> 팀 분할 seed 0~5 x 비율 0.4·0.6 (12개, 약 20분) -> 판정
# 표준 회복 모델(work/team_plan, work/team_plan2)은 이미 있는 것을 쓴다.
# 사용: nohup bash run_team_weighted.sh > work/team_weighted_queue.log 2>&1 &
# 기다리지 않고 바로 시작하려면: pkill -f wait_idle.py
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p work

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

exec 9> work/team_weighted.lock
if ! flock -n 9; then
  log "run_team_weighted.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행: seed 0, 비율 0.4, 16 step"
rm -rf work/team_weighted_smoke
python team_weighted.py train --seeds 0 --ratios 0.4 --steps 16 --out-name team_weighted_smoke \
  > work/team_weighted_smoke.log 2>&1
n=$(find work/team_weighted_smoke -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 1 ]; then
  log "시험 실행 실패 (결과 $n/1개). work/team_weighted_smoke.log를 확인하세요."
  exit 1
fi

log "본 실행: seed 0~5 x 비율 0.4·0.6"
python team_weighted.py train > work/team_weighted_train.log 2>&1
log "판정"
python team_weighted.py judge > work/team_weighted_result.txt 2>&1
log "전부 끝. 결과: work/team_weighted_result.txt"
