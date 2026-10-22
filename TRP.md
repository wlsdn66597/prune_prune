# TRP: 궤적 순위·회복 탐침 비균일 프루닝

Trajectory-ranked, Recovery-Probed pruning. 우리 실험에서 나온 발견을 설계 원칙으로 삼아, 기존 방법(NetAdapt)의 뼈대를 소수 클래스 전이 상황에 맞게 바꾼 프루닝 절차예요.

## 1. 출발점이 된 발견

| 발견 | 근거 (TrashNet, 22과제 × 5 fold) |
|---|---|
| D1. 자른 직후 점수는 최종 성능을 잘 예측하지 못한다 | 자른 직후 1등(Taylor, +4.9%p)이 회복 후에는 L2와 비슷함. 자른 직후 검증 정확도와 최종 정확도의 순위 상관 0.17, 56 step 회복 후 정확도와는 0.66 (비율 0.8) |
| D2. 많이 자를수록 파인튜닝 궤적(movement)이 가중치 크기(L2)보다 낫다 | L2 대비 +0.03 / +0.33 / +0.79 / +1.51 / +2.95%p (비율 0.2 / 0.4 / 0.6 / 0.8 / 0.9) |
| D3. 얼마나 줄일 수 있는지는 학습 전 유사도로 예측된다 | 실루엣 점수와 프루닝 여유(AUC)의 순위 상관 0.85. 클래스 수보다 설명력이 큼 |
| D4. 층마다 여유와 비용이 다르다 | layer4 블록 내부를 90% 자르면 파라미터는 66% 줄고 정확도는 거의 그대로. 대신 MACs는 20%만 줄어듦 |
| D5. 체크포인트 두 개로 movement를 근사할 수 없다 (AdamW) | 순위 상관 −0.13 |

## 2. 절차

입력: 사전학습 모델, 태스크 데이터, 자원 종류(파라미터 수 = 파일 크기, 또는 MACs)

0. **예산 계획 (D3).** 사전학습 특징의 클래스 간 실루엣 점수로, 정확도 하락을 3%p 안에 두면서 자를 수 있는 비율 r\*를 예측해요. 학습이 필요 없어요. → `predict_budget.py`
1. **파인튜닝하면서 movement 기록 (D2, D5).** 매 step마다 −grad × weight를 누적해요. 체크포인트로 사후에 근사할 수 없으니 학습 중에 기록해야 해요. → `train_base.py`
2. **비균일 배분 (D1, D4).** 목표 크기에 닿을 때까지 반복해요. → `trp_search.py`
   - 후보 단위는 DepGraph 그룹 12개예요. 블록 내부 채널 8개와 stage별 잔차 채널 4개예요.
   - 매 반복마다 단위마다 현재 자원의 20%가 줄도록 movement 순위가 낮은 채널을 지운 후보를 만들어요. 남는 채널이 4개 미만이 되는 단위는 건너뛰어요.
   - 후보마다 BN 통계를 다시 잡고, 고정된 학습 배치로 56 step 회복시킨 뒤 검증 NLL을 재요.
   - NLL이 가장 낮은 후보의 **구조만** 확정해요. 회복된 가중치는 버리고, 다음 반복은 원래 가중치에서 시작해요.
3. **최종 회복.** 예산마다 확정된 구조를 840 step 회복시켜요. 다른 비교 대상과 똑같은 양이에요.

## 3. 관련 연구와의 차이

| | NetAdapt (ECCV 2018) | EagleEye (ECCV 2020) | TRP (우리) |
|---|---|---|---|
| 층별 배분 | 반복마다 층별 후보, 가장 좋은 것 선택 | 무작위로 배분을 뽑아 평가 | NetAdapt와 같은 탐욕 방식. 단위는 DepGraph 그룹 |
| 층 안에서 지울 채널 | L2 크기가 작은 것 | L1 크기 | **파인튜닝 중 누적 movement가 작은 것** (D2) |
| 후보 평가 | 짧은 미세조정 후 정확도 | **BN 재보정만** (미세조정 없이) | **고정 배치로 56 step 회복 후 검증 NLL** (D1). 검증셋이 작아서 정확도보다 덜 흔들리는 NLL을 씀 |
| 고른 후보의 가중치 | 다음 반복으로 넘김 (계속 학습됨) | - | **넘기지 않고 구조만 확정.** 최종 모델의 학습량을 비교 대상과 같게 맞추고, 구조 선택과 학습 효과를 분리 |
| 예산 정하기 | 사용자가 지정 | 사용자가 지정 | **학습 전 유사도로 예측** (D3) |
| 자원 | 기기 실측 지연시간·에너지 | FLOPs | 파라미터 수(파일 크기) 또는 MACs (D4) |
| 대상 상황 | ImageNet 모델의 모바일 이식 | ImageNet | 사전학습 모델을 소수 클래스 태스크로 특화한 뒤 크게 줄이는 상황 |

관련 결과와의 관계는 이래요.
- **Random Channel Pruning (CVPR 2022):** 기준보다 층별 구성이 더 중요하다고 했어요. 우리 D2는 많이 자를 때는 기준도 중요하다는 걸 보였어요. TRP는 둘을 함께 다뤄요.
- **PRACTISE (CVPR 2023):** 블록 단위에서 회복 가능성(recoverability)이라는 개념을 썼어요. TRP의 탐침은 이 개념을 채널 단위 배분에 적용한 거예요.
- **Movement Pruning (NeurIPS 2020):** NLP의 개별 가중치 단위였던 기준을 CNN 채널 그룹 단위로 옮겼어요.

**주장할 수 있는 것:** 새로운 탐색 알고리즘이 아니라, 소수 클래스 전이 상황에서 측정한 결과로 구성 요소(순위 기준, 후보 평가 방식, 가중치를 넘기지 않음, 예산 예측)를 정하고, 각 요소의 기여를 비교 실험으로 검증한 절차예요.

## 4. 비교 실험

예산은 파라미터 기준 0.36 / 0.16 / 0.04 / 0.01이에요. 균일 프루닝 비율 0.4 / 0.6 / 0.8 / 0.9와 크기가 같아서, 균일 결과는 이미 있는 것을 써요. 과제는 6클래스와 가장 어려운/쉬운 쌍·삼중 5개 × 5 fold예요.

| 방법 | 확인하는 것 |
|---|---|
| uniform_l2, uniform_mov | 균일 프루닝 대비 이득 (이미 있음) |
| global_l2, global_taylor | 표준 전역 프루닝 대비 이득 |
| trp_bn (탐침 = BN 재보정만) | D1: EagleEye식 평가로 바꾸면 나빠지는가 |
| trp_l2 (순위 = L2) | D2: movement 순위가 배분 안에서도 기여하는가 |
| **trp** | 최종 |
| predict_budget.py | D3: 유사도로 예측한 예산이 실제로 쓸 만한가 |

**성공 기준:**
- trp가 같은 예산에서 uniform_mov와 global_l2보다 높아요. 과제·fold별로 짝지어 비교해요.
- trp가 trp_bn과 trp_l2보다 높아야 각 구성 요소가 기여한다고 말할 수 있어요.
- 탐색 비용(후보 수, 시간)도 함께 보고해요.

## 5. 한계

- 데이터셋 하나(TrashNet), 모델 하나(ResNet-18)에서의 결과예요.
- 탐욕 방식이라 전역 최적은 보장하지 않아요.
- 검증셋이 작아서(쌍은 약 128장) 후보 평가에 노이즈가 있어요.
- 자원을 파라미터 수와 MACs로 쟀어요. 라즈베리파이 실측 지연시간은 아니에요.

## 참고문헌

- NetAdapt, ECCV 2018 — [링크](https://openaccess.thecvf.com/content_ECCV_2018/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.html)
- EagleEye, ECCV 2020 — [arXiv 2007.02491](https://arxiv.org/abs/2007.02491)
- Revisiting Random Channel Pruning, CVPR 2022 — [링크](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Revisiting_Random_Channel_Pruning_for_Neural_Network_Compression_CVPR_2022_paper.html)
- Practical Network Acceleration with Tiny Sets (PRACTISE), CVPR 2023 — [링크](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Practical_Network_Acceleration_With_Tiny_Sets_CVPR_2023_paper.html)
- Movement Pruning, NeurIPS 2020 — [arXiv 2005.07683](https://arxiv.org/abs/2005.07683)
- Pruning CNNs for Resource Efficient Inference (전이 상황의 Taylor), ICLR 2017 — [arXiv 1611.06440](https://arxiv.org/abs/1611.06440)
- DepGraph, CVPR 2023 — [링크](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html)
- Few-Class Arena, ICLR 2025 — [링크](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d1b51bc11920d812fbc1d41dadd4246e-Abstract-Conference.html)
- Rethinking the Value of Network Pruning, ICLR 2019 — [arXiv 1810.05270](https://arxiv.org/abs/1810.05270)
