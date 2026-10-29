"""팀 분할 6분류: 자를수록 어느 클래스가 무너지나 (원본 대 team_plan 모델, 클래스별 test 재현율 %).

계획이 6분류에서 2.7~4.4%p 낙관적이었던 이유가 trash 같은 소수·잡동사니 클래스 때문인지 본다.
team 분할 test는 클래스마다 장수가 달라서(trash 약 21장, 1장 = 약 4.8%p) 장수도 같이 낸다.

사용: python team_classwise.py > work/team_classwise.txt
"""
import argparse

import numpy as np
import pandas as pd
import torch

import common as C
import prune_common as P


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--model-dir', default='team_plan', help='work/ 아래 자른 모델 폴더')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    images, _ = C.load_meta(args.work_dir, args.data_root)
    te_idx, te_y = C.team_split(images)['test']
    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)

    rows = []
    for s in args.seeds:
        models = {0.0: P.load_base(runs / 'team' / f'seed{s}', len(C.CLASSES6), args.device)}
        for p in sorted((runs.parent / args.model_dir / f'seed{s}').glob('mov_r*.pt')):
            models[float(p.stem.split('_r')[1])] = torch.load(p, map_location=args.device, weights_only=False).eval()
        for r, m in models.items():
            pred = C.predict_logits(m, cache, te_idx).argmax(1).numpy()
            for c, name in enumerate(C.CLASSES6):
                hit = pred[te_y == c] == c
                rows.append({'seed': s, 'ratio': r, 'class': name, 'recall': 100 * hit.mean(), 'n': int(len(hit))})
            rows.append({'seed': s, 'ratio': r, 'class': '전체', 'recall': 100 * (pred == te_y).mean(), 'n': len(te_y)})
    d = pd.DataFrame(rows)
    tab = d.pivot_table(index='class', columns='ratio', values='recall').reindex(C.CLASSES6 + ['전체'])
    print('=== 팀 분할 6분류: 클래스별 test 재현율 (%, 시드 평균) ===')
    print(tab.assign(장수=d.groupby('class').n.first()).round(1).to_string())
    print('\n원본 대비 하락 (%p, 양수면 떨어짐)')
    print(tab.rsub(tab[0.0], axis=0).drop(columns=0.0).round(1).to_string())
    worst = tab.rsub(tab[0.0], axis=0).drop(index='전체').drop(columns=0.0)
    for r in worst.columns:
        print(f'  비율 {r:g}: 가장 많이 떨어진 클래스 {worst[r].idxmax()} ({worst[r].max():+.1f}%p)')
    sd = d[d['class'] == 'trash'].groupby('ratio').recall.std()
    print('\ntrash 재현율의 시드 간 sd: ' + ', '.join(f'{r:g} {v:.1f}' for r, v in sd.items() if not np.isnan(v)))


if __name__ == '__main__':
    main()
