#!/bin/bash
# 팀 분할 6분류 계획 검증 대기열: 대기 -> 계획(학습 전 예측을 work/team_plan.json에 저장) -> 시험 실행
#   -> 고른 비율과 바로 위 비율을 시드 0·1·2로 movement 자르기 + 2800 step 회복 -> 판정
# 요구 85%(주)·88%·90%(참고). 비율 2~4개 x 시드 3개 = 6~12개 모델, 약 30분.
# 사용: nohup bash run_team_plan.sh > work/team_plan_queue.log 2>&1 &
# 기다리지 않고 바로 시작하려면: pkill -f wait_idle.py
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p work

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

exec 9> work/team_plan.lock
if ! flock -n 9; then
  log "run_team_plan.sh가 이미 실행 중입니다."
  exit 1
fi

log "계획: 학습 전 예측을 저장 (GPU 안 씀)"
if ! python team_plan.py plan > work/team_plan_plan.txt 2>&1; then
  log "계획 실패. work/team_plan_plan.txt를 확인하세요."
  exit 1
fi
cat work/team_plan_plan.txt

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "시험 실행: 시드 0, 16 step"
rm -rf work/team_plan_smoke
python team_plan.py train --seeds 0 --steps 16 --out-name team_plan_smoke > work/team_plan_smoke.log 2>&1
n=$(find work/team_plan_smoke -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -lt 1 ]; then
  log "시험 실행 실패 (결과 없음). work/team_plan_smoke.log를 확인하세요."
  exit 1
fi

log "본 실행: 시드 0·1·2"
python team_plan.py train > work/team_plan_train.log 2>&1
log "판정"
python team_plan.py judge > work/team_plan_result.txt 2>&1
log "전부 끝. 결과: work/team_plan_result.txt, 모델: work/team_plan/seed*/mov_r*.pt"
