#!/bin/bash
# 약한 클래스 보호 채널 선택의 확인과 분해 대기열: 대기 -> 새 팀 원본(seed 6~10) -> 시험 실행
#   -> 4가지(표준, 보호-손실, 균등-손실, 보호-logit) x 시드 5개, 비율 0.4 (20개) -> 판정. 약 45분.
# 사용: nohup bash run_team_protect2.sh > work/team_protect2_queue.log 2>&1 &
# 기다리지 않고 바로 시작하려면: pkill -f wait_idle.py
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p work

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

exec 9> work/team_protect2.lock
if ! flock -n 9; then
  log "run_team_protect2.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "새 팀 분할 원본 (seed 6~10)"
python baseline_team.py --seeds 6 7 8 9 10 > work/team_base6_10.log 2>&1

log "시험 실행: seed 6, 보호-logit, 16 step"
rm -rf work/team_protect2_smoke
python team_protect2.py train --seeds 6 --arms logit --steps 16 --out-name team_protect2_smoke \
  > work/team_protect2_smoke.log 2>&1
n=$(find work/team_protect2_smoke -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 1 ]; then
  log "시험 실행 실패 (결과 $n/1개). work/team_protect2_smoke.log를 확인하세요."
  exit 1
fi

log "본 실행: 4가지 x seed 6~10"
python team_protect2.py train > work/team_protect2_train.log 2>&1
log "판정"
python team_protect2.py judge > work/team_protect2_result.txt 2>&1
log "전부 끝. 결과: work/team_protect2_result.txt"
