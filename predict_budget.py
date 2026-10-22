"""TRP 0단계 검증: 학습 전에 계산한 유사도(실루엣)로 과제별 프루닝 여유(r*)를 예측할 수 있는가.

과제 하나씩 빼고 나머지로 회귀식을 맞춰 예측한다 (leave-one-task-out). 학습 없이 기존 결과만 쓴다.
예측한 비율로 잘랐다면 정확도가 얼마나 떨어졌을지도 그 과제의 실제 곡선에서 보간해 확인한다.

필요: summarize_prune.py --recover-steps 840 이 만든 work/runs/prune_results_rec840.csv
사용: python predict_budget.py [--crit mov_cum] [--drop 3]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C

RATIOS = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]


def drop_curves(work, crit):
    d = pd.read_csv(C.runs_dir(work) / 'prune_results_rec840.csv')
    d = d[(d.crit == crit) & d.ratio.isin(RATIOS)]
    b = pd.read_csv(C.runs_dir(work) / 'base_results.csv')[['task', 'fold', 'test_acc']]
    d = d.merge(b.rename(columns={'test_acc': 'base'}), on=['task', 'fold'])
    d['drop'] = (d.base - d.acc) * 100
    c = d.groupby(['task', 'ratio'])['drop'].mean().unstack('ratio')[RATIOS]
    return c[c.notna().all(axis=1)]


def rstar(drops, limit):
    """하락이 limit(%p)를 처음 넘는 지점을 선형 보간한 비율."""
    x, y = np.r_[0.0, RATIOS], np.r_[0.0, drops]
    for i in range(1, len(x)):
        if y[i] > limit:
            return x[i - 1] + (limit - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
    return x[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--crit', default='mov_cum')
    ap.add_argument('--drop', type=float, default=3.0, help='허용 정확도 하락 (%p)')
    args = ap.parse_args()

    curves = drop_curves(args.work_dir, args.crit)
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    t = pd.DataFrame({'rstar': [rstar(curves.loc[k].values, args.drop) for k in curves.index]}, index=curves.index)
    t = t.join(sim[['k', 'silhouette']])

    preds = {'상수(다른 과제 평균)': [], '실루엣': [], '실루엣+클래스수': []}
    for task in t.index:
        tr = t.drop(index=task)
        x = t.loc[task]
        preds['상수(다른 과제 평균)'].append(tr.rstar.mean())
        a, b = np.polyfit(tr.silhouette, tr.rstar, 1)
        preds['실루엣'].append(a * x.silhouette + b)
        A = np.c_[np.ones(len(tr)), tr.silhouette, tr.k]
        coef, *_ = np.linalg.lstsq(A, tr.rstar.values, rcond=None)
        preds['실루엣+클래스수'].append(float(coef @ [1, x.silhouette, x.k]))

    print(f'기준 {args.crit}, 허용 하락 {args.drop}%p, 과제 {len(t)}개 (과제 하나씩 빼고 예측)\n')
    rows = []
    for name, p in preds.items():
        p = np.clip(np.array(p), 0.0, RATIOS[-1])
        real_drop = [np.interp(pi, np.r_[0.0, RATIOS], np.r_[0.0, curves.loc[k].values]) for pi, k in zip(p, t.index)]
        real_drop = np.array(real_drop)
        rows.append({'예측 방법': name, 'r* 평균오차': np.abs(p - t.rstar.values).mean(),
                     '순위상관': spearmanr(p, t.rstar.values).correlation,
                     '예측대로 잘랐을 때 하락 평균(%p)': real_drop.mean(),
                     f'하락 ≤{args.drop:g}%p 과제 비율': (real_drop <= args.drop).mean(),
                     f'하락 ≤{args.drop + 1:g}%p 과제 비율': (real_drop <= args.drop + 1).mean()})
        t[f'pred_{name}'] = p
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('\n== 과제별 (실루엣 회귀 예측) ==')
    print(t[['k', 'silhouette', 'rstar', 'pred_실루엣']].sort_values(['k', 'silhouette']).round(3).to_string())


if __name__ == '__main__':
    main()
