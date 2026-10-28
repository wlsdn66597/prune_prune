"""예산 분할 실험 사후 분석: split_points.csv와 similarity.csv만 읽는다 (학습·GPU 없음, 몇 초).

split_eval.py 결과(H5 기각, 고정 비율로 충분)를 본 뒤 추가한 분석이다. 판정 기준은 아래 숫자들을 보기 전에 정했다.
  [A] 증폭     : 자를수록 4비트가 비싸지나. 조건(과제·fold)마다 W8−W4 손실을 비율별로 본다
  [B] 지배     : 비율 r의 W4 대, 다음 비율에서 그 크기 이하인 가장 높은 비트 (조건마다 짝 비교)
  [C] 환율     : 크기를 절반으로 줄이는 데 드는 정확도. 비트 단계(W8→mix6→mix5→W4)와 다음 비율로 자르기(W8)를 나란히
  [D] 고정 조합: 예산마다 모든 과제에 같은 (비율, 비트)를 쓰는 규칙(그 과제를 뺀 나머지의 평균 검증으로 고름) 대
                 조건마다 검증으로 고르기. 팀원에게 넘길 예산별 조합 표도 낸다
  [E] 난이도   : 과제별 프루닝 비용, 4비트 비용, 둘의 비, 원본 대비 3%p 안에서 가장 작은 크기를 실루엣과 비교
판정 (미리 정함):
  증폭 : 비율 0.8의 W8−W4 손실이 비율 0보다 크고 Wilcoxon p<0.01, 조건별 기울기(손실 ~ 비율)도 양수이고 p<0.01
  H6   : 실루엣과 log(3%p 안 최소 크기)의 Spearman ρ<0, p<0.05 -> '얼마나 줄일지는 과제가 정한다'
  H7   : 프루닝 비용과 4비트 비용이 둘 다 실루엣과 ρ<0, p<0.05이고 log(4비트 비용/프루닝 비용)은 p≥0.05
         -> '어떻게 나눌지는 과제가 정하지 않는다'(H5 기각의 이유). p≥0.05가 관계 없음의 증명은 아니다
주의: W8 이하는 모두 양자화 뒤 BN 재보정을 했고 fp32만 안 했다. 그래서 비트끼리의 비교는 W8 이하에서만 한다.
      [E]의 최소 크기는 test로 고른 값이라 낙관적이다 (H2의 r*와 같은 방식).

사용: python split_followup.py > work/split_followup.txt
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
from summarize_budget import fmt, stat

BUDGETS = [16, 8, 4, 2, 1, 0.5, 0.25, 0.1]  # split_eval.py와 같다
ORDER = ['fp32', 'W8', 'mix6', 'mix5', 'W4']  # 비트가 높은 순
TOL = 3  # [E] 원본 대비 허용 하락(%p), H2의 r*와 같다


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    r, p = spearmanr(x, y)
    return r, p


def rho_text(x, y):
    r, p = rho(x, y)
    return '-' if np.isnan(r) else f'{r:+.2f} (p={p:.3f}, n={len(x)})'


def amplification(acc, ratios):
    pen = acc.xs('W8', axis=1, level='bits') - acc.xs('W4', axis=1, level='bits')  # 조건 x 비율
    r0 = ratios[0]
    print(f'\n[A] 증폭: 4비트 손실(W8 − W4, %p)이 비율에 따라 커지나 (짝 비교는 같은 조건의 비율 {r0:g} 대비)')
    for r in ratios:
        line = f'  비율 {r:<4g} 평균 {pen[r].mean():5.2f} (sd {pen[r].std():4.2f})'
        if r != r0:
            line += f'  비율 {r0:g} 대비 {fmt(*stat((pen[r] - pen[r0]).dropna()))}'
        print(line)
    full = pen[ratios].dropna()
    slopes = full.apply(lambda s: np.polyfit(ratios, s.to_numpy(), 1)[0], axis=1)
    m, wtl, p = stat(slopes)
    print(f'  조건별 기울기 (%p / 비율 1.0): {fmt(m, wtl, p)}')
    if r0 != 0 or 0.8 not in ratios:
        return None
    m8, _, p8 = stat((pen[0.8] - pen[r0]).dropna())
    return m8 > 0 and p8 < 0.01 and m > 0 and p < 0.01


def dominance(acc, size, ratios):
    print('\n[B] 지배: 비율 r의 W4 대, 다음 비율에서 그 크기 이하인 가장 높은 비트 (조건별 짝, 다음 쪽 − W4)')
    for r, r2 in zip(ratios[:-1], ratios[1:]):
        d, used = [], []
        for i in acc.index:
            lim = size.loc[i, (r, 'W4')]
            fit = [b for b in ORDER if size.loc[i, (r2, b)] <= lim]
            if fit and not np.isnan(acc.loc[i, (r, 'W4')]):
                d.append(acc.loc[i, (r2, fit[0])] - acc.loc[i, (r, 'W4')])
                used.append(fit[0])
        if d:
            b2 = max(set(used), key=used.count)
            print(f'  {r:g} W4 ({size[(r, "W4")].mean():.3f}MB) 대 {r2:g} {b2} ({size[(r2, b2)].mean():.3f}MB): '
                  f'{fmt(*stat(d))}')


def exchange(acc, size, ratios):
    print('\n[C] 환율: 크기를 절반으로 줄이는 데 드는 정확도 (%p, 조건 평균, 작을수록 싸다)')
    rows = {}
    for i, r in enumerate(ratios):
        row = {f'{b1}→{b2}': ((acc[(r, b1)] - acc[(r, b2)]) / np.log2(size[(r, b1)] / size[(r, b2)])).mean()
               for b1, b2 in zip(ORDER[1:-1], ORDER[2:])}
        if i + 1 < len(ratios):
            r2 = ratios[i + 1]
            row['다음 비율로 자르기(W8)'] = ((acc[(r, 'W8')] - acc[(r2, 'W8')])
                                      / np.log2(size[(r, 'W8')] / size[(r2, 'W8')])).mean()
        rows[f'비율 {r:g}'] = row
    print(pd.DataFrame(rows).T.round(2).to_string())


def fixed_rules(df):
    print('\n[D] 예산별 규칙 (test %, 같은 조건끼리 짝)')
    print('    고정: 모든 과제에 같은 (비율, 비트). 그 과제를 뺀 나머지 과제들의 평균 검증으로 고른다')
    print('    조건별: 조건마다 예산 안에서 검증 최고 (split_eval [2]의 과제별 선택)')
    conds = df[['task', 'fold']].drop_duplicates()
    lookup = []
    for B in BUDGETS:
        f = df[df.size_mb <= B]
        if f.empty:
            continue
        own = (f.sort_values(['val', 'size_mb'], ascending=[False, True])
               .groupby(['task', 'fold']).head(1).set_index(['task', 'fold']).test)
        fixed, picks = {}, []
        for t in f.task.unique():
            g = f[f.task != t].groupby(['ratio', 'bits'])
            ranked = g.val.mean()[g.size() == (conds.task != t).sum()].sort_values(ascending=False)
            for fo in f[f.task == t].fold.unique():
                mine = f[(f.task == t) & (f.fold == fo)].set_index(['ratio', 'bits']).test
                k = next((k for k in ranked.index if k in mine.index), None)
                if k is not None:
                    fixed[(t, fo)] = mine[k]
                    picks.append(k)
        if not picks:
            print(f'  {B:>5g}MB  모든 과제에서 예산 안에 드는 조합이 없음')
            continue
        fixed = pd.Series(fixed)
        top = max(set(picks), key=picks.count)
        d = (fixed - own.reindex(fixed.index)).dropna()
        print(f'  {B:>5g}MB  고정 {top[0]:g}+{top[1]} ({picks.count(top)}/{len(picks)}): {fixed.mean():.2f}  '
              f'조건별: {own.mean():.2f}  고정 − 조건별 {fmt(*stat(d))}')
        g = f.groupby(['ratio', 'bits'])
        everywhere = g.val.mean()[g.size() == len(conds)]
        if not everywhere.empty:
            r, b = everywhere.idxmax()
            sel = f[(f.ratio == r) & (f.bits == b)]
            lookup.append(f'  {B:>5g}MB  비율 {r:g} + {b:<4}  크기 {sel.size_mb.mean():.3f}MB  test {sel.test.mean():.2f}')
    print('\n    팀 전달용 예산별 조합 (모든 과제의 평균 검증으로 고름)')
    print('\n'.join(lookup))


def difficulty(acc, size, ratios, sil):
    print('\n[E] 난이도: 과제별(fold 평균) 값과 실루엣의 순위 상관 (실루엣이 클수록 쉬운 과제)')
    ta, ts = acc.groupby(level='task').mean(), size.groupby(level='task').mean()
    base = ta[(ratios[0], 'fp32')]
    tasks = base.index
    prune = pd.concat([base - ta[(r, 'fp32')] for r in ratios[1:]], axis=1).mean(axis=1)
    quant = (ta.xs('W8', axis=1, level='bits') - ta.xs('W4', axis=1, level='bits')).mean(axis=1)
    smallest = ts.where(ta.ge(base - TOL, axis=0)).min(axis=1)
    s = sil.reindex(tasks)
    kind = np.asarray(tasks.str.split('_').str[0])
    items = {'prune': ('프루닝 비용 (fp32 하락, 자른 비율들 평균)', prune),
             'quant': ('4비트 비용 (W8 − W4, 모든 비율 평균)', quant),
             'share': ('log(4비트 비용 / 프루닝 비용)', np.log(quant.clip(lower=0.05) / prune.clip(lower=0.05))),
             'small': (f'log(원본 대비 {TOL}%p 안 최소 크기)', np.log(smallest))}
    res = {}
    for key, (name, y) in items.items():
        y = y.reindex(tasks)
        res[key] = rho(s, y)
        print(f'  {name}\n      전체 {rho_text(s, y)} | 쌍 {rho_text(s[kind == "pair"], y[kind == "pair"])} | '
              f'삼중 {rho_text(s[kind == "tri"], y[kind == "tri"])}')
    tab = pd.DataFrame({'실루엣': s, '원본': base, '프루닝비용': prune, '4비트비용': quant,
                        f'{TOL}%p안최소MB': smallest})
    print(tab.sort_values('실루엣').round(3).to_string())
    neg = {k: res[k][0] < 0 and res[k][1] < 0.05 for k in res}
    return neg['small'], neg['prune'] and neg['quant'] and not res['share'][1] < 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    df = pd.read_csv(C.runs_dir(args.work_dir) / 'split_points.csv')
    sil = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task').silhouette
    acc = df.pivot_table(index=['task', 'fold'], columns=['ratio', 'bits'], values='test')
    size = df.pivot_table(index=['task', 'fold'], columns=['ratio', 'bits'], values='size_mb')
    ratios = sorted(df.ratio.unique())
    print(f'=== 예산 분할 사후 분석 (조건 {len(acc)}개, test 정확도 %) ===')
    amp = amplification(acc, ratios)
    dominance(acc, size, ratios)
    exchange(acc, size, ratios)
    fixed_rules(df)
    h6, h7 = difficulty(acc, size, ratios, sil)
    print('\n[판정]')
    if amp is not None:
        print(f'  증폭: {"확인" if amp else "확인 안 됨"} (비율 0.8 대 0의 4비트 손실 차이와 조건별 기울기, 둘 다 p<0.01 기준)')
    print(f'  H6 (얼마나 줄일지는 과제가 정한다): {"지지" if h6 else "지지 안 됨"}')
    print(f'  H7 (두 비용이 난이도와 함께 움직이고, 둘의 비는 난이도와 무관): {"지지" if h7 else "지지 안 됨"}')


if __name__ == '__main__':
    main()
