"""배분 + 점진 결합 실험 요약: 같은 구조(목표 배분)에서 나눠 자르기 vs 한 번에 자르기, 갱신 vs 고정, 일정 비교.

비교 대상(이미 있는 결과)
  oneshot_trp_bn  : runs/trp 의 trp_bn (TRP 배분, 한 번에 자르고 840 step 회복)
  oneshot_joint   : runs/joint 의 trp_joint (TRP-J 배분, 한 번에)
  oneshot_uniform : prune_grid rec840 의 mov_cum (균일, 한 번에)
  gtp_uniform_e60 : runs/gradual 의 gtp (균일, 나눠 자르기, 60% 지점 도달, 갱신)
사용: python summarize_gradalloc.py
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

RATIO_OF_BUDGET = {0.36: 0.4, 0.16: 0.6, 0.04: 0.8, 0.01: 0.9}
PAIRS = [
    ('grad_trp_online_e30', 'oneshot_trp_bn', 'TRP 배분: 나눠 자르기(30%) vs 한 번에'),
    ('grad_trp_online_e60', 'oneshot_trp_bn', 'TRP 배분: 나눠 자르기(60%) vs 한 번에'),
    ('grad_trp_online_e30', 'grad_trp_online_e60', '일정: 30% 도달 vs 60% 도달 (TRP 배분)'),
    ('grad_trp_online_e30', 'grad_trp_fixed_e30', 'movement 갱신 vs 고정 (TRP 배분)'),
    ('grad_uniform_online_e30', 'oneshot_uniform', '균일: 나눠 자르기(30%) vs 한 번에'),
    ('grad_uniform_online_e30', 'gtp_uniform_e60', '일정: 30% 도달 vs 60% 도달 (균일, 기존 GTP)'),
    ('grad_joint_online_e30', 'oneshot_joint', 'TRP-J 배분: 나눠 자르기(30%) vs 한 번에'),
    ('grad_joint_online_e30', 'oneshot_uniform', 'TRP-J 배분 + 나눠 자르기 vs 균일 한 번에 (같은 크기·연산량 이하)'),
]


def rows_from(paths, method_fn, ratio_fn):
    rows = []
    for f in paths:
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'task': r['task'], 'fold': r['fold'], 'seed': r['seed'], 'method': method_fn(r),
                     'ratio': ratio_fn(r), 'acc': r['test']['acc'], 'params_ratio': r['params_ratio'],
                     'macs_ratio': r['macs_ratio']})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--out-name', default='gradalloc')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    runs = C.runs_dir(args.work_dir)

    rows = rows_from([p for p in (runs / args.out_name).rglob('*.json') if re.search(r'_r[\d.]+\.json$', p.name)],
                     lambda r: r['method'], lambda r: r['ratio'])
    if not rows:
        raise SystemExit('결과가 없습니다.')
    rows += rows_from(list((runs / 'trp').rglob('trp_bn_b*.json')), lambda r: 'oneshot_trp_bn',
                      lambda r: RATIO_OF_BUDGET.get(r['budget']))
    rows += rows_from(list((runs / 'joint').rglob('trp_joint_r*.json')), lambda r: 'oneshot_joint',
                      lambda r: r['ratio'])
    rows += rows_from(list((runs / 'gradual').rglob('gtp_r*.json')), lambda r: 'gtp_uniform_e60',
                      lambda r: r['ratio'])
    df = pd.DataFrame(rows)
    u = pd.read_csv(runs / 'prune_results_rec840.csv')
    u = u[u.crit == 'mov_cum'].assign(method='oneshot_uniform')
    df = pd.concat([df, u[['task', 'fold', 'seed', 'method', 'ratio', 'acc', 'params_ratio', 'macs_ratio']]],
                   ignore_index=True)
    mine = df[df.method.str.startswith('grad_')]
    keys = mine[['task', 'fold', 'seed']].drop_duplicates()
    df = df.merge(keys, on=['task', 'fold', 'seed'])
    df = df[df.ratio.isin(mine.ratio.unique())]
    df.to_csv(runs / 'gradalloc_results.csv', index=False)
    print(f'과제·fold {len(keys)}개, 균일 비율 {sorted(mine.ratio.unique())} (TRP 예산은 같은 크기로 맞춤)\n')

    for v, name in (('acc', 'test 정확도'), ('params_ratio', '남은 파라미터'), ('macs_ratio', '남은 MACs')):
        print(f'== {name}: 방법 × 균일 비율 ==')
        print(df.pivot_table(index='method', columns='ratio', values=v, aggfunc='mean').round(4).to_string(), '\n')

    key = ['task', 'fold', 'seed', 'ratio']
    for a, b, title in PAIRS:
        x = df[df.method == a].set_index(key)['acc']
        y = df[df.method == b].set_index(key)['acc']
        d = ((x - y).dropna() * 100).rename('diff').reset_index()
        if d.empty:
            continue
        print(f'== {title}: {a} − {b} (%p) ==')
        for r, g in d.groupby('ratio'):
            nz = g['diff'][g['diff'] != 0]
            p = wilcoxon(nz).pvalue if len(nz) > 5 else np.nan
            print(f"ratio {r}: 평균 {g['diff'].mean():+.2f}, 이긴 비율 {(g['diff'] > 0).mean():.2f}, "
                  f"n={len(g)}, Wilcoxon p={p:.3f}")
        print()


if __name__ == '__main__':
    main()
