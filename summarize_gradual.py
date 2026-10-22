"""GTP 결과 요약: 같은 비율·같은 학습량에서 점진(gradual) vs 한 번에(one-shot), 순위 기준별 비교.

one-shot(oneshot_mov, oneshot_l2)은 summarize_prune.py --recover-steps 840 이 만든 CSV에서 가져온다.
사용: python summarize_gradual.py
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import common as C

PAIRS = [('gtp', 'oneshot_mov'), ('gtp_fixed', 'oneshot_mov'), ('gtp', 'gtp_fixed'),
         ('gtp_l2', 'oneshot_l2'), ('gtp', 'gtp_l2'), ('oneshot_mov', 'oneshot_l2')]


def load(work, out_name):
    rows = []
    for f in (C.runs_dir(work) / out_name).rglob('*.json'):
        if not re.search(r'_r[\d.]+\.json$', f.name):
            continue
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'task': r['task'], 'fold': r['fold'], 'seed': r['seed'], 'method': r['method'],
                     'ratio': r['ratio'], 'acc': r['test']['acc'], 'nll': r['test']['nll'],
                     'params_ratio': r['params_ratio']})
    g = pd.DataFrame(rows)
    if g.empty:
        raise SystemExit('점진 프루닝 결과가 없습니다.')
    o = pd.read_csv(C.runs_dir(work) / 'prune_results_rec840.csv')
    o = o[o.crit.isin(['l2', 'mov_cum']) & o.ratio.isin(g.ratio.unique())].copy()
    o['method'] = o.crit.map({'l2': 'oneshot_l2', 'mov_cum': 'oneshot_mov'})
    o = o.merge(g[['task', 'fold', 'seed']].drop_duplicates(), on=['task', 'fold', 'seed'])
    return pd.concat([g, o[['task', 'fold', 'seed', 'method', 'ratio', 'acc', 'nll', 'params_ratio']]],
                     ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--out-name', default='gradual')
    args = ap.parse_args()
    pd.set_option('display.width', 200)

    df = load(args.work_dir, args.out_name)
    df.to_csv(C.runs_dir(args.work_dir) / 'gradual_results.csv', index=False)
    print(f"과제 {df.task.nunique()}개, 과제·fold {df[['task', 'fold']].drop_duplicates().shape[0]}개\n")
    for v in ('acc', 'nll', 'params_ratio'):
        print(f'== test {v}: 방법 × 비율 (평균) ==' if v != 'params_ratio' else '== 남은 파라미터 비율 ==')
        print(df.pivot_table(index='method', columns='ratio', values=v, aggfunc='mean').round(4).to_string(), '\n')

    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    key = ['task', 'fold', 'seed', 'ratio']
    for a, b in PAIRS:
        x = df[df.method == a].set_index(key)['acc']
        y = df[df.method == b].set_index(key)['acc']
        d = ((x - y).dropna() * 100).rename('diff').reset_index()
        if d.empty:
            continue
        print(f'== {a} − {b} (%p) ==')
        for r, g in d.groupby('ratio'):
            tm = g.groupby('task')['diff'].mean()
            p = wilcoxon(tm).pvalue if len(tm) > 5 and (tm != 0).any() else np.nan
            line = (f'ratio {r}: 평균 {g["diff"].mean():+.2f}, 이긴 비율 {(g["diff"] > 0).mean():.2f}, '
                    f'과제 {len(tm)}개 Wilcoxon p={p:.3f}')
            j = tm.to_frame().join(sim[['k', 'silhouette']])
            for k in (2, 3):
                jk = j[j.k == k]
                if len(jk) >= 5:
                    line += f', 유사도와 상관(k={k}) {spearmanr(jk.silhouette, jk["diff"]).correlation:+.2f}'
            print(line)
        print()


if __name__ == '__main__':
    main()
