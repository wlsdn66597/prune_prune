"""TRP 결과 요약: 같은 파라미터 예산에서 방법별 정확도, 균일 프루닝 대비 차이, 층별 남은 채널, 탐색 비용.

균일 프루닝(uniform_l2, uniform_mov)은 prune_grid.py rec840 결과(summarize_prune.py가 만든 CSV)에서 가져온다.
사용: python summarize_trp.py
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

UNIFORM = {  # 균일 비율 -> 같은 자원의 예산
    'params': {0.4: 0.36, 0.6: 0.16, 0.8: 0.04, 0.9: 0.01},
    'macs': {0.4: 0.372, 0.6: 0.173, 0.8: 0.048, 0.9: 0.015},
}
WIDTH = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}


def load_trp(work, out_name):
    rows = []
    for f in (C.runs_dir(work) / out_name).rglob('*.json'):
        if not re.search(r'_b[\d.]+\.json$', f.name):
            continue
        r = json.loads(f.read_text(encoding='utf-8'))
        ch = r['channels']
        row = {'task': r['task'], 'fold': r['fold'], 'seed': r['seed'], 'method': r['method'],
               'budget': r['budget'], 'acc': r['test']['acc'], 'nll': r['test']['nll'],
               'params_ratio': r['params_ratio'], 'macs_ratio': r['macs_ratio'],
               'search_sec': r.get('search_sec', np.nan), 'n_probes': r.get('n_probes', np.nan)}
        for s, w in WIDTH.items():  # 남은 채널 비율: 블록 내부(두 블록 평균), 잔차 채널
            row[f'{s}_in'] = (ch[f'{s}.0.conv1'] + ch[f'{s}.1.conv1']) / (2 * w)
            row[f'{s}_res'] = ch[f'{s}.0.conv2'] / w
        rows.append(row)
    return pd.DataFrame(rows)


def load_uniform(work, keys, resource):
    f = C.runs_dir(work) / 'prune_results_rec840.csv'
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d = d[d.crit.isin(['l2', 'mov_cum']) & d.ratio.isin(list(UNIFORM[resource]))].copy()
    d['method'] = d.crit.map({'l2': 'uniform_l2', 'mov_cum': 'uniform_mov'})
    d['budget'] = d.ratio.map(UNIFORM[resource])
    d = d.merge(keys, on=['task', 'fold', 'seed'])
    return d[['task', 'fold', 'seed', 'method', 'budget', 'acc', 'nll', 'params_ratio', 'macs_ratio']]


def paired(df, a, b):
    """같은 (과제, fold, 예산)끼리 짝지은 a - b (%p), a가 이긴 비율, Wilcoxon p."""
    k = ['task', 'fold', 'seed', 'budget']
    x = df[df.method == a].set_index(k)['acc']
    y = df[df.method == b].set_index(k)['acc']
    d = (x - y).dropna() * 100
    rows = []
    for bud, g in d.groupby(level='budget'):
        p = wilcoxon(g).pvalue if (g != 0).any() and len(g) > 5 else np.nan
        rows.append({'budget': bud, 'n': len(g), 'diff_%p': g.mean(), 'win': (g > 0).mean(), 'p': p})
    return pd.DataFrame(rows).set_index('budget')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--out-name', default='trp')
    ap.add_argument('--resource', choices=['params', 'macs'], default='params',
                    help='예산 기준. macs면 균일 프루닝을 같은 MACs끼리 짝짓는다')
    args = ap.parse_args()
    pd.set_option('display.width', 220)

    trp = load_trp(args.work_dir, args.out_name)
    if trp.empty:
        raise SystemExit('TRP 결과가 없습니다.')
    keys = trp[['task', 'fold', 'seed']].drop_duplicates()
    df = pd.concat([trp, load_uniform(args.work_dir, keys, args.resource)], ignore_index=True)
    df.to_csv(C.runs_dir(args.work_dir) / f'{args.out_name}_results.csv', index=False)
    print(f'예산 기준: {args.resource}')
    print(f'과제 {sorted(keys.task.unique())}, 과제·fold {len(keys)}개\n')

    unit = '남는 파라미터 비율' if args.resource == 'params' else '남는 MACs 비율'
    print(f'== test 정확도 (과제·fold 평균): 방법 × 예산({unit}) ==')
    print(df.pivot_table(index='method', columns='budget', values='acc', aggfunc='mean').round(4).to_string())
    print('\n== 실제로 남은 파라미터 / MACs 비율 ==')
    print(df.pivot_table(index='method', columns='budget', values=['params_ratio', 'macs_ratio'],
                         aggfunc='mean').round(3).to_string())

    for ref in ('uniform_mov', 'global_l2'):
        if ref not in set(df.method):
            continue
        for m in sorted(set(df.method) - {ref}):
            r = paired(df, m, ref)
            if not r.empty:
                print(f'\n== {m} − {ref} ==')
                print(r.round(3).to_string())

    cols = [f'{s}_{t}' for s in WIDTH for t in ('in', 'res')]
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')['silhouette']
    t = trp[trp.method == 'trp'].copy()
    if not t.empty:
        t['silhouette'] = t.task.map(sim)
        print('\n== TRP가 남긴 채널 비율 (stage별 블록 내부 / 잔차), 과제 × 예산 ==')
        print(t.groupby(['budget', 'task'])[cols].mean().sort_index(ascending=[False, True]).round(2).to_string())

    print('\n== 탐색 비용 (기준 모델 하나당, 가장 작은 예산까지) ==')
    s = trp.groupby('method')[['search_sec', 'n_probes']].max()
    print(s.round(1).to_string())


if __name__ == '__main__':
    main()
