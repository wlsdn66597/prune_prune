"""기준 x 회복 예산: 회복을 840에서 2800 step으로 늘려도 기준 간 격차가 남는지 본다 (prune_grid 결과).

잡음 확인 실험에서 균일(mov_cum) 모델은 회복을 늘리면 크게 올랐다(0.9에서 최대 +10%p). 그래서 840 step에서 본
기준 간 격차가 '어느 쪽이 더 빨리 회복되나'일 수 있다. 같은 과제·fold끼리 짝지어 두 예산의 격차를 비교한다.

판정 규칙 (결과를 보기 전에 정함):
  유지: 2800 step에서 mov_cum - L2가 비율 0.8이나 0.9 중 하나 이상에서 +1%p 이상이고 Wilcoxon p < 0.05
  소멸: 2800 step에서 mov_cum - L2가 0.8과 0.9 모두 +0.5%p 미만
  그 외: 중간 (회복 예산이 빠듯할수록 기준이 중요)

사용: python summarize_budget.py
      python summarize_budget.py --budgets 56 840 --ratios 0.8   # 이미 있는 결과로 짧은 예산 쪽 경향 미리 보기
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

GROUPS = {'pair': 'pair', 'tri': 'triple', 'all6': 'multi', 'mat5': 'multi'}


def stat(d):
    """평균 차이(%p), 승/무/패, Wilcoxon p."""
    d = np.asarray(d, dtype=float)
    p = wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan
    w, l = int((d > 0).sum()), int((d < 0).sum())
    return d.mean(), f'{w}/{len(d) - w - l}/{l}', p


def fmt(m, wtl, p):
    return f'{m:+6.2f} [{wtl:>8}] p={p:.3f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--out-name', default='prune')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--criteria', nargs='+', default=['mov_cum', 'l2', 'random'])
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.8, 0.9])
    ap.add_argument('--budgets', type=int, nargs='+', default=[840, 2800], help='짧은 예산, 긴 예산 순서')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    short, long = args.budgets[0], args.budgets[-1]

    rows = []
    root = C.runs_dir(args.work_dir) / args.out_name
    for steps in args.budgets:
        for f in root.rglob(f'*_rec{steps}.json'):
            r = json.loads(f.read_text(encoding='utf-8'))
            if r['fold'] in args.folds and r['criterion'] in args.criteria and r['ratio'] in args.ratios:
                rows.append({'task': r['task'], 'fold': r['fold'], 'ratio': r['ratio'], 'crit': r['criterion'],
                             'steps': steps, 'acc': 100 * r['test']['acc']})
    if not rows:
        raise SystemExit('결과가 없습니다.')
    wide = pd.DataFrame(rows).pivot_table(index=['task', 'fold', 'ratio'], columns=['crit', 'steps'], values='acc')
    need = [(c, s) for c in args.criteria for s in args.budgets]
    lacking = [c for c in need if c not in wide.columns]
    if lacking:
        raise SystemExit(f'결과가 하나도 없는 조합: {lacking}')
    full = wide[need].dropna()
    print(f'=== 기준 x 회복 예산 (test 정확도, fold {args.folds}) ===')
    print(f'모든 기준·예산이 끝난 조건 {len(full)}개 / 결과가 있는 조건 {len(wide)}개')

    verdict = {}
    for ratio in args.ratios:
        if ratio not in full.index.get_level_values('ratio'):
            continue
        w = full.xs(ratio, level='ratio')
        print(f'\n== 비율 {ratio}: 조건 {len(w)}개 ==')
        acc = pd.DataFrame({s: [w[(c, s)].mean() for c in args.criteria] for s in args.budgets}, index=args.criteria)
        acc['gain'] = acc[long] - acc[short]
        print(f'test 정확도 평균 (%), gain = {long} - {short} step')
        print(acc.round(2).to_string())
        print('\n격차 (%p, 같은 과제·fold끼리 짝): 평균 [승/무/패] Wilcoxon p, 변화 = 긴 예산 격차 - 짧은 예산 격차')
        for a, b in (('mov_cum', 'l2'), ('random', 'l2'), ('mov_cum', 'random')):
            if a not in args.criteria or b not in args.criteria:
                continue
            gs, gl = w[(a, short)] - w[(b, short)], w[(a, long)] - w[(b, long)]
            print(f'  {a + "-" + b:<15}{short}: {fmt(*stat(gs))}   {long}: {fmt(*stat(gl))}   '
                  f'변화: {fmt(*stat(gl - gs))}')
            if (a, b) == ('mov_cum', 'l2'):
                verdict[ratio] = stat(gl)
                group = w.index.get_level_values('task').str.split('_').str[0].map(GROUPS)
                by = pd.DataFrame({short: gs.to_numpy(), long: gl.to_numpy()}, index=group)
                mov_l2 = by.groupby(level=0).mean().round(2)
                mov_l2['n'] = by.groupby(level=0).size()
        if 'mov_cum' in args.criteria and 'l2' in args.criteria:
            print('\nmov_cum-l2 과제 묶음별 (pair=2클래스, triple=3, multi=5~6)')
            print(mov_l2.to_string())

    if verdict:
        print('\n== 판정 (규칙은 이 파일 맨 위 설명에 있음) ==')
        for ratio, (m, wtl, p) in verdict.items():
            print(f'  비율 {ratio}: {long} step에서 mov_cum-l2 = {m:+.2f}%p [{wtl}] p={p:.3f}')
        keep = any(m >= 1.0 and p < 0.05 for m, _, p in verdict.values())
        gone = all(m < 0.5 for m, _, _ in verdict.values())
        print('  => ' + ('(임시) ' if len(full) < len(wide) else '')
              + ('유지: 회복을 늘려도 movement 기준의 이점이 남는다' if keep else
                 '소멸: 회복을 늘리면 기준 간 격차가 사라진다' if gone else
                 '중간: 회복 예산이 빠듯할수록 기준이 중요하다'))


if __name__ == '__main__':
    main()
