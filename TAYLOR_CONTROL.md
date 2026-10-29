# Taylor 혼합 대조군

## 질문과 비교

기존 movement 대비 약 2%p 개선이 단순한 Taylor 혼합 효과인지, 클래스별 중요도 집계 효과인지 분리한다.

- C: movement + 클래스별 Taylor를 표준화한 뒤 균등 집계.
- E: movement + 같은 probe의 부호 있는 기울기를 모두 합친 뒤 Taylor 계산.
- 두 방법 모두 `z(movement) + z(Taylor)`이며, 전체/전역 채널 프루닝을 뜻하지 않는다.
- 동일한 클래스별 표본, 증강, forward/backward를 한 번만 수행해 C/E 점수를 동시에 만든다. E도 동일한 층화 표본을 사용한다. 따라서 모집단의 자연 클래스 비율을 따르는 별도 표본 실험은 아니다.
- 기존 코드와 동일하게 배치 평균 CE 기울기를 누적하며, 마지막 작은 배치도 기존 코드의 가중 방식과 같다.
- 시드 6~10, 채널 제거 비율 0.4, BN 재보정, 회복 2800 step. 동일 시드에서는 회복 데이터 순서/증강도 맞춘다.
- 기존 C는 중요도 측정 직전 증강 RNG를 고정하지 않았으므로, 공정한 주 비교를 위해 새 C도 함께 학습한다. 본 실행은 10모델이다.

## 실행 전 고정한 판정

주 비교는 **새 C − 새 E**. 전체 정확도 평균 +1.0%p 이상, 5개 시드 중 4개 이상 승리하면 이 조건에서 클래스별 집계의 추가 효과를 뒷받침한다. 미충족이면 근거가 부족한 것이며, 동등성이나 Taylor 혼합만이 원인임을 증명하지 않는다.

기존 A/B/C는 참고용으로만 함께 출력한다. 이 대조군은 실루엣 가중의 필요성을 증명하는 실험이 아니다. 프루닝 비율과 파라미터/MACs 및 채널 수 일치를 검사하고, 결과가 부족하면 판정을 중단한다. 검증/테스트 데이터는 중요도 계산이나 방법 선택에 사용하지 않는다.

## 서버에서 실행

저장소에 파일이 반영된 뒤, 기존 GPU 서버 터미널에서:

```bash
cd ~/prune_prune
git pull --ff-only
mkdir -p work
nohup bash run_team_taylor_control.sh > work/team_taylor_control_queue.log 2>&1 &
```

대기열은 입력/기존 체크포인트와 CUDA를 검사하고, 기존 실험이 끝날 때까지 기다린다. 16-step 시험 실행이 성공하면 본 실행을 시작한다. 기존 프로세스 중단, 원본 모델 재학습, 기존 결과 삭제는 하지 않는다. 같은 대기열의 중복 실행은 lock으로 차단한다.

진행 상황:

```bash
tail -n 30 work/team_taylor_control_queue.log
tail -n 30 work/team_taylor_control_train.log
```

완료 결과:

```bash
cat work/team_taylor_control_result.txt
```

수치 파일은 `work/team_taylor_control/per_seed_metrics.csv`, `paired_C_minus_E.csv`, `summary.json`에 저장된다. 재실행 시 같은 프로토콜로 완료된 모델은 건너뛴다.
