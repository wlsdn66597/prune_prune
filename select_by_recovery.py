"""회복 가능성 기반 선택 (추가 실험, 학습 없음 - 저장된 결과만 분석).

기준 후보(random, l2, bn, taylor, mov_ckpt, mov_cum) 중 하나를 모델마다 고를 때
  - 자른 직후(BN 재보정) 검증 정확도로 고르기   : EagleEye(ECCV 2020) 방식
  - 짧은 회복(기본 56 step) 후 검증 정확도로 고르기 : 회복 가능성 기반
중 어느 쪽이 최종(840 step 회복 후) test 정확도가 높은 후보를 더 잘 고르는지 비교한다.

필요한 결과: prune_grid.py의 rec0, rec56(--ratios 0.6 0.8), rec840
사용: python select_by_recovery.py
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import common as C


def load(work, out_name, recs):
    rows = []
    for f in (C.runs_dir(work) / out_name).rglob('*.json'):
        m = re.search(r'_rec(\d+)\.json$', f.name)
        if not m or int(m.group(1)) not in recs:
            continue
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'task': r['task'], 'fold': r['fold'], 'seed': r['seed'], 'crit': r['criterion'],
                     'ratio': r['ratio'], 'rec': int(m.group(1)), 'val_acc': r['val']['acc'],
                     'val_nll': r['val']['nll'], 'test_acc': r['test']['acc']})
    return pd.DataFrame(rows)


def pick(g, col_acc, col_nll):
    """검증 정확도가 가장 높은 후보, 동점이면 검증 NLL이 낮은 후보."""
    return g.sort_values([col_acc, col_nll], ascending=[False, True]).iloc[0]['crit']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--out-name', default='prune')
    ap.add_argument('--short', type=int, default=56, help='짧은 회복 step')
    ap.add_argument('--final', type=int, default=840, help='최종 회복 step')
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.6, 0.8])
    args = ap.parse_args()
    pd.set_option('display.width', 200)

    df = load(args.work_dir, args.out_name, {0, args.short, args.final})
    df = df[df.ratio.isin(args.ratios)]
    key = ['task', 'fold', 'seed', 'ratio', 'crit']
    w = df.pivot_table(index=key, columns='rec', values=['val_acc', 'val_nll', 'test_acc'])
    w.columns = [f'{a}_{b}' for a, b in w.columns]
    need = ['val_acc_0', f'val_acc_{args.short}', f'test_acc_{args.final}']
    w = w.dropna(subset=need).reset_index()
    if w.empty:
        raise SystemExit(f'rec0, rec{args.short}, rec{args.final} 결과가 모두 있는 후보가 없습니다.')
    final = f'test_acc_{args.final}'

    rows, rhos = [], []
    for (t, f, s, r), g in w.groupby(['task', 'fold', 'seed', 'ratio']):
        if g['crit'].nunique() < 3:
            continue
        acc = g.set_index('crit')[final]
        res = {'task': t, 'fold': f, 'seed': s, 'ratio': r, 'n_cand': len(g),
               'select_immediate': acc[pick(g, 'val_acc_0', 'val_nll_0')],
               'select_short': acc[pick(g, f'val_acc_{args.short}', f'val_nll_{args.short}')],
               'oracle_test': acc.max()}
        if f'val_acc_{args.final}' in g:  # 최종 예산으로 전부 회복시킨 뒤 검증으로 고르기 (비싼 참고값)
            res['select_full'] = acc[pick(g, f'val_acc_{args.final}', f'val_nll_{args.final}')]
        for c in acc.index:
            res[f'fixed_{c}'] = acc[c]
        rows.append(res)
        rhos.append({'ratio': r,
                     'rho_immediate': spearmanr(g['val_acc_0'], g[final]).correlation,
                     'rho_short': spearmanr(g[f'val_acc_{args.short}'], g[final]).correlation})
    res = pd.DataFrame(rows)
    print(f'비교한 (과제, fold, 비율) {len(res)}개, 후보 수 {sorted(res.n_cand.unique())}\n')

    cols = [c for c in res.columns if c.startswith(('select_', 'fixed_', 'oracle'))]
    print(f'== 최종({args.final} step 회복 후) test 정확도 평균 ==')
    print(res.groupby('ratio')[cols].mean().T.round(4).to_string())

    print('\n== 선택 점수와 최종 정확도의 순위 상관 (후보 간 Spearman, 평균) ==')
    print(pd.DataFrame(rhos).groupby('ratio').mean().round(3).to_string())

    print('\n== 짧은 회복 선택 vs 자른 직후 선택 (과제별 fold 평균 후 Wilcoxon) ==')
    for r, g in res.groupby('ratio'):
        t = g.groupby('task')[['select_short', 'select_immediate']].mean()
        d = (t['select_short'] - t['select_immediate']) * 100
        p = wilcoxon(d).pvalue if np.any(d != 0) else float('nan')
        print(f'ratio {r}: 차이 {d.mean():+.2f}%p, 짧은 회복이 나은 과제 {int((d > 0).sum())}/{len(d)}, p={p:.3f}')
    res.to_csv(C.runs_dir(args.work_dir) / 'select_by_recovery.csv', index=False)


if __name__ == '__main__':
    main()
