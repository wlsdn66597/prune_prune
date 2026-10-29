#!/bin/bash
# 6분류 적용을 위한 과제군 넓히기 대기열:
#   대기 -> 클래스별 재현율(1판 모델) -> 과제 25개 추가 -> 시험 실행 -> 새 팀 분할 원본(seed 3·4·5)
#   -> 새 과제 기준 모델(fold 0·1) -> 새 과제 movement 2800 step(비율 0.4~0.9) -> 2판 계획(학습 전 예측 저장)
#   -> 2판 모델(seed 3·4·5) -> 판정
# 예상: 기준 모델 약 15분 + 프루닝 250개 약 2.5시간 + 팀 분할 약 30분.
# 사용: nohup bash run_team_ext.sh > work/team_ext_queue.log 2>&1 &
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

exec 9> work/team_ext.lock
if ! flock -n 9; then
  log "run_team_ext.sh가 이미 실행 중입니다."
  exit 1
fi

log "다른 학습이 끝나기를 기다리는 중 (실험 스크립트, run_*.sh 대기열, GPU 사용률을 1분마다 확인)"
python wait_idle.py

log "클래스별 재현율 (1판 모델, 몇 분)"
python team_classwise.py > work/team_classwise.txt 2>&1 || log "클래스별 재현율 실패 (계속 진행). work/team_classwise.txt 확인"

log "과제 25개 추가"
if ! python add_tasks.py > work/add_tasks.txt 2>&1; then
  log "과제 추가 실패. work/add_tasks.txt를 확인하세요."
  exit 1
fi
EXT=$(cat work/meta/tasks_ext.txt)

log "시험 실행: 새 과제 기준 모델과 자르기 (pair_cb-tr, fold 0, 16 step)"
rm -rf work/runs/smoke_ext work/runs/smoke_ext_prune
python train_base.py --tasks pair_cb-tr --folds 0 --total-steps 16 --eval-every 8 --out-name smoke_ext \
  --workers 4 > work/smoke_ext.log 2>&1
python prune_grid.py --tasks pair_cb-tr --folds 0 --criteria mov_cum --ratios 0.4 --recover-steps 16 \
  --base-name smoke_ext --out-name smoke_ext_prune --workers 4 >> work/smoke_ext.log 2>&1
n=$(find work/runs/smoke_ext_prune -name '*.json' 2>/dev/null | wc -l)
if [ "$n" -ne 1 ]; then
  log "시험 실행 실패 (결과 $n/1개). work/smoke_ext.log를 확인하세요."
  exit 1
fi

log "새 팀 분할 원본 (seed 3·4·5)"
python baseline_team.py --seeds 3 4 5 > work/team_base345.log 2>&1
log "새 과제 기준 모델 (25과제 x fold 0·1, 프로세스 3개)"
shards ext_base python train_base.py --tasks "$EXT" --folds 0 1
log "새 과제 movement 자르기 (비율 0.4~0.9, 2800 step, 프로세스 3개)"
shards ext_prune python prune_grid.py --tasks "$EXT" --folds 0 1 --criteria mov_cum \
  --ratios 0.4 0.6 0.7 0.8 0.9 --recover-steps 2800

log "2판 계획: 학습 전 예측 저장"
if ! python team_plan.py plan --family ext --plan-file team_plan2.json > work/team_plan2_plan.txt 2>&1; then
  log "계획 실패. work/team_plan2_plan.txt를 확인하세요."
  exit 1
fi
cat work/team_plan2_plan.txt
log "2판 모델 (seed 3·4·5)"
python team_plan.py train --plan-file team_plan2.json --seeds 3 4 5 --out-name team_plan2 > work/team_plan2_train.log 2>&1
python team_plan.py judge --plan-file team_plan2.json --seeds 3 4 5 --out-name team_plan2 > work/team_plan2_result.txt 2>&1
log "전부 끝. 결과: work/team_classwise.txt, work/team_plan2_plan.txt, work/team_plan2_result.txt"
