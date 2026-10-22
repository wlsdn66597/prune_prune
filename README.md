# 프루닝 연구 코드 (0~2단계)

전처리와 학습 설정은 `compare_fc.ipynb`의 B 모델(fc를 6클래스로 교체)과 같아요.

| 파일 | 단계 | 하는 일 |
|---|---|---|
| `common.py` | - | 공통 코드: 전처리, 분할, 학습 루프, movement 기록 |
| `prepare_data.py` | 0-A | 이미지 목록, 평가용 캐시, 사전학습 특징, 중복 후보 리포트 |
| `make_folds.py` | 0-B | 중복 그룹, 이미지별 fold, 재질 클래스당 400장, 과제 22개, 과제별 유사도 |
| `baseline_team.py` | 1 | 팀 분할에서 노트북 결과 재현 (seed 0, 1, 2) |
| `train_base.py` | 2 | 22과제 × 5 fold 기준 모델 학습 + movement 기록 |
| `summarize.py` | - | 결과 표 |

## 설치

1. PyTorch 공식 사이트의 설치 명령으로 CUDA 버전에 맞는 `torch`, `torchvision`을 먼저 설치해요.
2. `pip install -r requirements.txt`

## 실행 순서

모든 명령은 이 폴더에서 실행하고, 결과는 `work/` 아래에 쌓여요.

### 0-A. 데이터 준비

```bash
python prepare_data.py --data-root <dataset-resized 경로>
```

- `--data-root`에는 cardboard, glass 등 클래스 폴더 6개가 바로 들어 있는 폴더를 넣어요.
- 확인할 것
  - 클래스별 장수가 403 / 501 / 410 / 594 / 482 / 137인지
  - `work/meta/dup_sheet_*.png`를 열어서, 같은 물체를 다시 찍은 사진이 나오기 시작하는 유사도 구간을 찾아요. 그 구간의 아래쪽 값이 다음 단계의 `--dup-thr`예요.

### 0-B. fold 고정과 과제 만들기

```bash
python make_folds.py --dup-thr 0.95
```

- 같은 물체 사진이 없다고 판단되면 `--dup-thr 1.01`로 실행해요(그룹을 만들지 않음).
- 4클래스 조합 5개까지 넣으려면 `--with-quads`를 붙여요.
- 확인할 것
  - fold별 클래스 장수가 고르게 나뉘었는지
  - 가장 큰 그룹이 너무 크지 않은지
  - 유사도 표에서 유리-플라스틱, 골판지-종이 같은 쌍이 어려운 쪽(silhouette가 낮은 쪽)에 오는지

### 시험 실행 (몇 분)

본 실행 전에 에러가 없는지 짧게 확인해요.

```bash
python baseline_team.py --epochs 1 --seeds 0 --out-name smoke_team
python train_base.py --tasks pair_gl-pl --folds 0 --total-steps 20 --eval-every 10 --out-name smoke
```

`work/runs/smoke/pair_gl-pl/fold0_seed0/`에 `model.pt`, `movement.pt`, `logits.npz`, `result.json` 네 파일이 생기면 정상이에요. `smoke`로 시작하는 폴더는 확인 후 지워도 돼요.

### 1단계. 팀 분할 재현

```bash
python baseline_team.py
```

- 확인할 것: test acc가 노트북 결과(0.914 ± 0.013) 근처인지. GPU와 워커 수가 달라서 똑같이 나오지는 않아요. 2%p 안쪽이면 정상이에요.
- `work/runs/team/`과 `work/meta/team_split.json`을 팀에 공유해요.

### 2단계. 기준 모델 110개

```bash
python train_base.py                         # 한 프로세스로 전부
# 또는 터미널 3개에서 동시에
python train_base.py --shard 0 --nshards 3
python train_base.py --shard 1 --nshards 3
python train_base.py --shard 2 --nshards 3

python summarize.py
```

- 끝난 실험은 건너뛰니까, 중간에 끊겨도 같은 명령으로 다시 실행하면 이어서 돌아요.
- 확인할 것
  - `summarize.py` 표의 과제별 정확도
  - `best_step`이 항상 맨 끝(840)이나 맨 앞(56)에 몰리지 않는지
  - 실행당 시간(`sec`)

## 출력 구조

```
work/
  meta/   images.csv, tasks.json, similarity.csv, team_split.json,
          eval224_uint8.pt, feat_pretrained.npy, dup_nn.csv, dup_*.png
  runs/
    team/seed0/                  model.pt, movement.pt, logits.npz, result.json
    base/<과제>/fold0_seed0/     (같은 네 파일)
```

`movement.pt` 구성 (conv 이름별 dict)

| 키 | 내용 |
|---|---|
| `cum_out[name]`, `cum_in[name]` | 평가 시점마다의 누적 movement(-grad × weight)를 출력/입력 채널별로 합한 값. 모양 `[평가 시점 수, 채널 수]` |
| `best_i` | 고른 체크포인트의 평가 시점 번호 |
| `ckpt_out[name]`, `ckpt_in[name]` | 체크포인트 근사 sum(w_best² − w_pre²)의 채널별 합 |

- 고른 체크포인트에 맞는 누적값은 `cum_out[name][best_i]`예요.
- 첫 평가 구간을 뺀 값은 `cum_out[name][best_i] - cum_out[name][0]`이에요. 새로 붙인 헤드가 무작위인 초반 구간을 빼고 싶을 때 써요.

## 노트북과 다른 점

- 평가 이미지는 Resize → CenterCrop까지 한 번만 계산해 캐시하고, 나머지(/255와 정규화)는 GPU에서 해요. 계산 결과는 같아요.
- epoch 대신 step으로 세요. 팀 분할은 1 epoch가 56 step이라 840 step이 15 epoch와 정확히 같아요. 연구 과제들은 데이터 크기와 상관없이 840 step으로 업데이트 횟수를 맞춰요.
- 학습 중에 movement를 기록해요. 기울기를 읽기만 해서 학습 결과에는 영향이 없어요.
- 혼합 정밀도(AMP)는 쓰지 않았어요. 결과가 노트북과 달라질 수 있고, 이 크기에서는 데이터 로딩이 병목이라 속도 이득도 작아요.

## 윈도우에서 돌릴 때

- 주피터 안에서 실행하면 DataLoader 워커가 깨질 수 있으니 터미널에서 `.py`로 실행하세요.
- 워커 시작이 느리면 `--workers 4` 정도로 줄여 보세요.

## 용량

`model.pt`가 실행당 약 45MB라 2단계까지 합쳐 5GB 정도, 평가 캐시가 약 380MB예요.
