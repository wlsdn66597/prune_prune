#!/bin/bash
# 6분류 계획 3판 대기열: 대기 -> 1단계(척도 확인, 학습 없음) -> 통과하면: 새 팀 원본(seed 6~10)
#   -> 3판 계획(가장 약한 클래스의 실루엣, 학습 전 예측 저장) -> 3판 모델(seed 6~10) -> 판정
# 1단계를 통과하지 못하면 거기서 멈춘다. 2단계까지 가면 약 40분.
# 사용: nohup bash run_team_v3.sh > work/team_v3_queue.log 2>&1 &
# 기다리지 않고 바로 시작하려면: pkill -f wait_idle.py
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p work

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

exec 9> work/team_v3.lock
if ! flock -n 9; then
  log "run_team_v3.sh가 이미 실행 중입니다."
  exit 1
fi

log "1단계: 척도 확인 (학습 없음, 몇 분)"
if ! python feature_check.py > work/feature_check.txt 2>&1; then
  log "1단계 실패. work/feature_check.txt를 확인하세요."
  exit 1
fi
cat work/feature_check.txt
if [ ! -f work/feature_check_pass ]; then
  log "1단계를 통과하지 못해 3판은 여기서 멈춥니다."
  exit 0
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py
log "새 팀 분할 원본 (seed 6~10)"
python baseline_team.py --seeds 6 7 8 9 10 > work/team_base6_10.log 2>&1
log "3판 계획: 학습 전 예측 저장"
if ! python team_plan.py plan --family ext --feature minclass --plan-file team_plan3.json \
    > work/team_plan3_plan.txt 2>&1; then
  log "계획 실패. work/team_plan3_plan.txt를 확인하세요."
  exit 1
fi
cat work/team_plan3_plan.txt
log "3판 모델 (seed 6~10)"
python team_plan.py train --plan-file team_plan3.json --seeds 6 7 8 9 10 --out-name team_plan3 \
  > work/team_plan3_train.log 2>&1
python team_plan.py judge --plan-file team_plan3.json --seeds 6 7 8 9 10 --out-name team_plan3 \
  > work/team_plan3_result.txt 2>&1
log "전부 끝. 결과: work/team_plan3_result.txt"
