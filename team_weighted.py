"""분리도 가중 회복: 학습 전에 무너질 클래스를 알고, 회복 학습에서 그 클래스를 지킨다 (팀 분할 6분류).

근거 (class_drop.py, 276조건): 과제 안에서 사전학습 특징의 클래스별 실루엣이 낮은 클래스일수록 프루닝 뒤 재현율이
더 떨어진다(순위 상관 −0.38). 파인튜닝 뒤 재현율로는 알 수 없다(−0.03). 팀 분할에서 metal은 실루엣이 가장 낮고
(−0.017) 원본 재현율은 95%였는데, 비율 0.4에서 −10.8%p로 가장 먼저 무너졌다.
방법: 팀 분할 학습 이미지의 클래스별 실루엣 s_c로 회복 손실 가중치를 정한다 (학습 전에 정해지고 추가 비용 없음).
  z_c = (평균 s - s_c) / sd(s),  w_c = clip(1 + 0.5 z_c, 0.5, 2),  평균이 1이 되게 나눈다 (값은 하나로 고정, 고르지 않음)
비교: 같은 원본(seed 0~5)을 같은 movement로 자르고 2800 step 회복한 표준 모델(team_plan, team_plan2에 이미 있음).
판정 (결과를 보기 전에 정함): 가장 약한 클래스 = 실루엣이 가장 낮은 클래스(학습 전에 정해짐).
  비율마다, 시드 6개 평균으로 그 클래스 재현율이 3%p 이상 오르고 전체 정확도가 0.5%p 넘게 떨어지지 않으면 '효과 있음'.
  0.4와 0.6 둘 다면 '효과 있음', 하나면 '부분', 없으면 '효과 없음'. 클래스별 재현율과 macro F1도 같이 낸다.

사용: python team_weighted.py train && python team_weighted.py judge > work/team_weighted_result.txt
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, silhouette_samples

import common as C
import prune_common as P
from plan_eval import REC_STEPS

RATIOS = [0.4, 0.6]
SEEDS = [0, 1, 2, 3, 4, 5]
BASELINE = {s: ('team_plan' if s < 3 else 'team_plan2') for s in SEEDS}  # 표준 회복 모델 폴더 (work/ 아래)


def class_weights(work_dir, data_root):
    """팀 분할 학습 이미지의 클래스별 실루엣과 회복 가중치."""
    images, _ = C.load_meta(work_dir, data_root)
    idx, y = C.team_split(images)['train']
    feats = np.load(C.meta_dir(work_dir) / 'feat_pretrained.npy')
    s = silhouette_samples(feats[idx], y, metric='cosine')
    sil = np.array([s[y == c].mean() for c in range(len(C.CLASSES6))])
    w = np.clip(1 + 0.5 * (sil.mean() - sil) / sil.std(), 0.5, 2.0)
    return sil, w / w.mean()


def train(args, work):
    sil, w = class_weights(args.work_dir, args.data_root)
    print('클래스 실루엣: ' + ', '.join(f'{c} {v:+.3f}' for c, v in zip(C.CLASSES6, sil)))
    print('회복 가중치  : ' + ', '.join(f'{c} {v:.2f}' for c, v in zip(C.CLASSES6, w)))
    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
    train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for s in args.seeds:
        base_dir = runs / 'team' / f'seed{s}'
        base, move = P.load_base(base_dir, len(C.CLASSES6), args.device), P.load_movement(base_dir)
        for r in args.ratios:
            out = work / args.out_name / f'seed{s}' / f'mov_r{r}.json'
            if out.exists():
                continue
            t0 = time.time()
            C.set_seed(s)
            model = copy.deepcopy(base)
            P.prune(model, P.make_importance('mov_cum', model, move), r)
            P.recalibrate_bn(model, cache, tr_idx)
            P.recover(model, train_ds, args.steps, args.device, args.workers, s, class_weight=w)
            info = {'seed': s, 'ratio': r, 'steps': args.steps, 'class_weight': w.tolist(),
                    'val': C.metrics(C.predict_logits(model, cache, va_idx), va_y),
                    'test': C.metrics(C.predict_logits(model, cache, te_idx), te_y), 'time_sec': round(time.time() - t0, 1)}
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu(), out.with_suffix('.pt'))
            out.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
            print(f"seed{s} 비율 {r}: test {100 * info['test']['acc']:.2f}% ({info['time_sec']}s)", flush=True)
            del model


def judge(args, work):
    sil, w = class_weights(args.work_dir, args.data_root)
    weak = C.CLASSES6[int(np.argmin(sil))]
    images, _ = C.load_meta(args.work_dir, args.data_root)
    te_idx, te_y = C.team_split(images)['test']
    cache = C.EvalCache(args.work_dir, args.device)
    rows = []
    for s in args.seeds:
        for r in args.ratios:
            for arm, folder in [('표준', BASELINE[s]), ('가중', args.out_name)]:
                p = work / folder / f'seed{s}' / f'mov_r{r}.pt'
                if not p.exists():
                    print(f'모델 없음, 건너뜀: {p}')
                    continue
                m = torch.load(p, map_location=args.device, weights_only=False).eval()
                pred = C.predict_logits(m, cache, te_idx).argmax(1).numpy()
                row = {'seed': s, 'ratio': r, 'arm': arm, '전체': 100 * (pred == te_y).mean(),
                       'macroF1': 100 * f1_score(te_y, pred, average='macro')}
                row.update({c: 100 * (pred[te_y == i] == i).mean() for i, c in enumerate(C.CLASSES6)})
                rows.append(row)
    d = pd.DataFrame(rows)
    cols = ['전체', 'macroF1', *C.CLASSES6]
    print(f'=== 분리도 가중 회복 (팀 분할 6분류, 시드 {args.seeds}, test %) ===')
    print('가중치: ' + ', '.join(f'{c} {v:.2f}' for c, v in zip(C.CLASSES6, w)) + f'   가장 약한 클래스(실루엣 최저): {weak}')
    verdicts = []
    for r in args.ratios:
        t = d[d.ratio == r].pivot_table(index='seed', columns='arm', values=cols)
        diff = {c: t[c]['가중'] - t[c]['표준'] for c in cols}
        print(f'\n비율 {r:g} (시드 평균, 가중 − 표준, 가중이 나은 시드 수)')
        for c in cols:
            dc = diff[c].dropna()
            print(f'  {c:<10} 표준 {t[c]["표준"].mean():6.2f}  가중 {t[c]["가중"].mean():6.2f}  '
                  f'차이 {dc.mean():+6.2f}  ({int((dc > 0).sum())}/{len(dc)})')
        ok = diff[weak].mean() >= 3 and diff['전체'].mean() >= -0.5
        verdicts.append(ok)
        print(f'  -> {weak} 재현율 {diff[weak].mean():+.2f}%p, 전체 {diff["전체"].mean():+.2f}%p: {"충족" if ok else "미충족"}')
    print('\n[판정] ' + {2: '효과 있음', 1: '부분', 0: '효과 없음'}[sum(verdicts)]
          + f' (비율 {", ".join(f"{r:g}" for r, v in zip(args.ratios, verdicts) if v) or "없음"}에서 충족)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['train', 'judge'])
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--ratios', type=float, nargs='+', default=RATIOS)
    ap.add_argument('--steps', type=int, default=REC_STEPS)
    ap.add_argument('--out-name', default='team_weighted', help='가중 회복 모델 폴더 (work/ 아래)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    work = C.runs_dir(args.work_dir).parent
    {'train': train, 'judge': judge}[args.step](args, work)


if __name__ == '__main__':
    main()
