# GTP: 점진 궤적 프루닝

Gradual Trajectory Pruning. [TRP](TRP.md)가 "어디를 자를지(층별 배분)"를 다뤘다면, GTP는 **"언제 자를지(점진 일정)"**와 **"순위를 언제 갱신할지"**를 다뤄요. 팀 파이프라인의 반복 프루닝(프루닝 ⇄ 1차 PCR)이 실제로 한 번에 자르는 것보다 나은지에 대한 근거도 돼요.

## 1. 출발점

| 우리 발견 | GTP에서의 쓰임 |
|---|---|
| D1. 자른 직후 상태만 보고 한 번에 정하면 틀리기 쉽다 | 결정을 한 번에 내리지 않고, 학습하면서 나눠서 내려요 |
| D2. 파인튜닝 궤적(movement)이 많이 자를수록 L2보다 낫다 | 기준 모델의 movement에서 시작하고, 자른 뒤 회복하는 궤적도 계속 쌓아 순위를 갱신해요 |
| D5. 체크포인트로 사후 근사할 수 없다 | 학습 루프 안에서 매 step 바로 누적해요 |

## 2. 절차

전체 학습량은 840 step으로, one-shot 회복과 같아요.

1. 기준 모델(파인튜닝 + movement 기록)에서 시작해요.
2. 56 step마다 모든 그룹을 같은 비율로 조금씩 잘라요. 비율은 cubic 일정 s(t) = s_f · (1 − (1 − t/t_end)³)를 따르고, 504 step(전체의 60%)에서 목표 비율에 도달해요. 자를 때마다 9번에 걸쳐 잘라요.
3. 자를 때의 순위는 "기준 모델의 movement + 지금까지 회복 중에 쌓인 movement"예요.
4. 목표 비율에 도달한 뒤 남은 336 step은 구조를 고정하고 학습해요. 학습률은 one-shot과 같은 cosine 일정(1e-4 → 0)이에요.

## 3. 관련 연구와의 차이

| | GMP (Zhu & Gupta 2017) | Movement Pruning (NeurIPS 2020) | GTP |
|---|---|---|---|
| 단위 | 개별 가중치 | 개별 가중치 | **채널 그룹** (DepGraph, 실제로 작아짐) |
| 순위 | 지금의 가중치 크기 | 프루닝하는 학습 중 누적 movement | **파인튜닝 때 쌓은 movement + 회복 중 계속 누적** |
| 일정 | cubic | cubic | cubic (같음) |
| 대상 | 처음부터 학습 | 사전학습 BERT의 파인튜닝 | 사전학습 CNN을 소수 클래스 태스크로 특화한 뒤 |

같은 질문을 다룬 벤치마크(NeurIPS 2024 ML and Compression 워크숍)는 "한 번에 자르기는 낮은 비율에서, 반복 자르기는 높은 비율에서 낫다"고 보고했어요. 우리 실험은 이 경향이 소수 클래스 전이 상황에서도 성립하는지 비율 0.6 / 0.8 / 0.9에서 확인해요.

## 4. 비교 실험

과제는 22개 × 5 fold이고, 비율은 0.6 / 0.8 / 0.9예요. 모든 방법의 최종 채널 수와 학습량이 같아요.

| 비교 | 확인하는 것 |
|---|---|
| gtp − oneshot_mov | 같은 기준(movement)에서 점진이 한 번에 자르기보다 나은가 |
| gtp − gtp_fixed | 회복 중에 movement를 계속 갱신하는 게 기여하는가 |
| gtp_l2 − oneshot_l2 | 크기 기준에서도 점진이 나은가 (채널 단위 GMP) |
| gtp − gtp_l2 | 점진 안에서도 movement가 L2보다 나은가 |
| oneshot_mov − oneshot_l2 | 기존 결과 재확인 (점검용) |

**성공 기준:** gtp가 oneshot_mov보다 높고(과제 22개 Wilcoxon), gtp_fixed보다도 높아야 "회복 궤적까지 반영하는 점진 프루닝"이 기여한다고 말할 수 있어요.

## 5. 한계

- 자를 때마다 옵티마이저(AdamW)를 다시 만들어서 모멘트가 초기화돼요.
- 일정(56 step 간격, 60% 지점 도달)은 한 가지 설정만 시험해요.
- 모든 층을 같은 비율로 잘라요. TRP의 비균일 배분과 합치는 건 후속 과제예요.

## 참고문헌

- Zhu & Gupta, To prune, or not to prune (2017) — [arXiv 1710.01878](https://arxiv.org/abs/1710.01878)
- Movement Pruning, NeurIPS 2020 — [arXiv 2005.07683](https://arxiv.org/abs/2005.07683)
- How Many Does It Take to Prune a Network: Comparing One-Shot vs. Iterative Pruning Regimes, NeurIPS 2024 Workshop on Machine Learning and Compression — [링크](https://neurips.cc/virtual/2024/98213)
