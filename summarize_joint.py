"""TRP-J 결과 요약: 균일 모델과 파라미터·MACs가 같거나 작은 조건에서 정확도 비교.

사용: python summarize_joint.py
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C


def load(work, out_name):
    rows = []
    for f in (C.runs_dir(work) / out_name).rglob('*.json'):
        if not re.search(r'_r[\d.]+\.json$', f.name):
            continue
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'task': r['task'], 'fold': r['fold'], 'seed': r['seed'], 'method': r['method'],
                     'ratio': r['ratio'], 'acc': r['test']['acc'], 'nll': r['test']['nll'],
                     'params_ratio': r['params_ratio'], 'macs_ratio': r['macs_ratio'],
                     'chosen': r.get('chosen'), 'accepted': r.get('accepted_moves')})
    j = pd.DataFrame(rows)
    if j.empty:
        raise SystemExit('TRP-J 결과가 없습니다.')
    u = pd.read_csv(C.runs_dir(work) / 'prune_results_rec840.csv')
    u = u[u.crit.isin(['l2', 'mov_cum']) & u.ratio.isin(j.ratio.unique())].copy()
    u['method'] = u.crit.map({'l2': 'uniform_l2', 'mov_cum': 'uniform_mov'})
    u = u.merge(j[['task', 'fold', 'seed']].drop_duplicates(), on=['task', 'fold', 'seed'])
    return pd.concat([j, u[['task', 'fold', 'seed', 'method', 'ratio', 'acc', 'nll', 'params_ratio',
                            'macs_ratio']]], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--out-name', default='joint')
    args = ap.parse_args()
    pd.set_option('display.width', 200)

    df = load(args.work_dir, args.out_name)
    df.to_csv(C.runs_dir(args.work_dir) / 'joint_results.csv', index=False)
    print(f"과제 {sorted(df.task.unique())}, 과제·fold {df[['task', 'fold']].drop_duplicates().shape[0]}개")
    print('예산 = 같은 비율로 자른 균일 모델의 파라미터와 MACs (둘 다 넘지 않음)\n')
    for v, name in (('acc', 'test 정확도'), ('params_ratio', '남은 파라미터'), ('macs_ratio', '남은 MACs')):
        print(f'== {name}: 방법 × 균일 비율 ==')
        print(df.pivot_table(index='method', columns='ratio', values=v, aggfunc='mean').round(4).to_string(), '\n')

    key = ['task', 'fold', 'seed', 'ratio']
    for a, b in (('trp_joint', 'uniform_mov'), ('joint_greedy', 'uniform_mov'), ('trp_joint', 'joint_greedy')):
        x = df[df.method == a].set_index(key)['acc']
        y = df[df.method == b].set_index(key)['acc']
        d = ((x - y).dropna() * 100).rename('diff').reset_index()
        if d.empty:
            continue
        print(f'== {a} − {b} (%p) ==')
        for r, g in d.groupby('ratio'):
            nz = g['diff'][g['diff'] != 0]
            p = wilcoxon(nz).pvalue if len(nz) > 5 else np.nan
            print(f"ratio {r}: 평균 {g['diff'].mean():+.2f}, 이긴 비율 {(g['diff'] > 0).mean():.2f}, "
                  f"같음 {(g['diff'] == 0).mean():.2f}, n={len(g)}, Wilcoxon p={p:.3f}")
        print()

    t = df[df.method == 'trp_joint']
    print('== TRP-J가 최종으로 고른 구성 (비율별 횟수) ==')
    print(pd.crosstab(t.ratio, t.chosen).to_string())
    print('\n받아들인 재배분 수 평균:', t.groupby('ratio')['accepted'].mean().round(2).to_dict())


if __name__ == '__main__':
    main()
