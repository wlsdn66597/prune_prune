"""과제별 계획 진단 (학습 없음, 몇 초): 계획 검증(plan_eval.py) 결과를 본 뒤 정한 탐색 단계다.
fold 0·1로 탐색하고, 통과한 항목만 fold 2·3·4로 확인한다. 판정 기준은 아래 숫자들을 보기 전에 정했다.

  [A] 배포 요구 목표: test 정확도 90%(주)·92%(보조) 이상을 지키며 최대한 자르기. 원본부터 못 미치는 조건은 빼고 센다
  [B] 원래 목표(원본 대비 하락 3%p 이하)를 같은 방식으로 다시
      결정 방식 (규칙은 모두 평가하는 과제를 빼고 만든다):
        고정       : 다른 과제 조건의 80% 이상이 목표를 지키는 가장 큰 비율 (모든 과제에 같은 비율)
        FCA식      : 실루엣으로 원본 정확도만 예측하고 하락은 다른 과제 평균 ([A]만. Few-Class Arena식 난이도 활용)
        곡선(k), 곡선(실루엣), 곡선(실루엣+k): 비율별 정확도([B]는 하락)를 그 특징으로 회귀해 예측 (우리 계획)
        탐색       : 모든 비율을 회복하고 검증으로 고름.  사후 최적: test로 고름 (상한)
      예측 방식은 비율을 0부터 올리며 '예측값 − 여유'가 목표를 지키는 데까지 고른다. 여유는 다른 과제들에서
      (그 과제도 빼고 예측해) 80% 이상 지키는 가장 작은 값이다.
      비교 지표 '이득': 같은 달성률에서 과제를 보지 않는 규칙보다 평균 비율을 얼마나 더 높이나. 과제를 보지 않는 규칙은
      모든 조건에 같은 비율을 쓰되 이웃한 두 비율을 섞을 수 있다고 보고, 비율별 달성률을 직선으로 이어 구한다
      (비율 후보가 성겨서 고정 규칙은 0.4에서 0.6으로 건너뛰므로, 달성률이 다른 규칙끼리 바로 비교하지 않는다).
      실행은 파인튜닝 + movement(주)이고, 파인튜닝 생략(사전학습 가중치 L2) 실행도 같은 절차로 참고로 낸다.
  [C] 파인튜닝 생략 손해(파인튜닝 + movement − 생략, 비율 0.4~0.9 평균)가 어려운 과제에 몰리나
  [D] 회복 이득(2800 − 840 step, movement)이 어려운 과제에 몰리나. 0.8·0.9는 같은 코드(prune_grid), 0.4·0.6은 참고
  [E] H2가 약해진 이유: 840·2800 step을 같은 비율 후보(0.4·0.6·0.8·0.9)로 맞춰 r*와 실루엣의 상관을 비교한다.
      840 step은 fold 5개와 fold 2개 조합 10개, 2800 step은 fold 0·1 (실루엣은 H2처럼 전체 이미지 기준)

판정 (미리 정함, 실행은 파인튜닝 + movement 기준):
  [A] 과제별 계획 이득: 곡선(실루엣)의 이득이 0.05 이상 (90%가 주, 92%는 보조)
      프루닝 쪽 기여  : 곡선(실루엣)의 이득이 FCA식의 이득보다 0.03 이상 크다
  [B] 원래 목표 회복  : 곡선(실루엣)의 이득이 0.05 이상
  ([A]·[B]는 처음에 '달성률 5%p 허용'으로 정했으나, 가짜 데이터 시험에서 조건 38개의 표본 잡음(약 ±6%p)과
   비율 후보의 성김 때문에 판정이 흔들려, 실제 결과를 보기 전에 위 지표로 바꿨다)
  [C] 실루엣과 생략 손해의 Spearman ρ<0, p<0.05 -> '쉬운 과제만 생략' 규칙이 의미 있다
  [D] 실루엣과 회복 이득(0.8·0.9)의 Spearman ρ<0, p<0.05 -> 난이도가 회복 속도를 정한다
  [E] fold 2개 조합들의 840 step ρ 평균이 2800 step ρ보다 0.2 이상 높으면 '회복 길이 때문', 아니면 'fold 수(잡음) 때문'

확인 단계 (fold 2·3·4, 탐색 결과를 본 뒤 확인 데이터를 만들기 전에 정함):
  파인튜닝 + movement 결과는 prune_grid(mov_cum, 2800 step)에서, 비율 0은 기준 모델 result.json에서 읽는다.
  확인 항목은 [A] 요구 90%의 과제별 계획 이득과 [D]이고, 기준은 탐색과 같다. 둘 다 '예'면 재현됐다고 본다.
  [A]의 프루닝 쪽 기여, [A] 92%, [B]는 참고로 낸다. 생략 실행을 학습하지 않았으면 [C]는 보류, [E]는 판정하지 않는다.
  2차 확인 (1차가 재현된 뒤, 생략 실행 fold 2·3·4를 학습하기 전에 정함): [C]와, 생략 실행 요구 90%에서
  곡선(실루엣)의 이득이 FCA식보다 0.03 이상 큰가(탐색 +0.30). 둘 다 '예'면 재현됐다고 본다.

사용: python plan_diag.py > work/plan_diag.txt                          # 탐색 (fold 0·1)
      python plan_diag.py --folds 2 3 4 > work/plan_confirm.txt         # 확인
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

import common as C
from plan_eval import GRID, REC_STEPS, load_points, rstar, train_silhouettes
from split_followup import rho_text
from summarize_budget import fmt, stat
from summarize_inherit import FT_STEPS

RELIABILITY = 0.8  # 고정 규칙과 여유 보정이 맞추는 달성률
MARGINS = np.arange(0, 8.01, 0.25)  # 여유 후보 (%p)
GRID840 = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]  # 840 step 곡선 실험(H2)의 비율
E_GRID = [0.0, 0.4, 0.6, 0.8, 0.9]  # [E]에서 840·2800 step이 같이 가진 비율
MODELS = {'곡선(k)': lambda s, k: [1.0, k], '곡선(실루엣)': lambda s, k: [1.0, s],
          '곡선(실루엣+k)': lambda s, k: [1.0, s, k]}
MAIN = '곡선(실루엣)'


def load_prune(runs, tag, steps):
    """prune_grid 결과: task, fold, ratio, val, test(%)."""
    rows = []
    for p in runs.glob(f'prune/*/fold*_seed0/{tag}_r*_rec{steps}.json'):
        j = json.loads(p.read_text(encoding='utf-8'))
        rows.append({'task': j['task'], 'fold': j['fold'], 'ratio': j['ratio'],
                     'val': 100 * j['val']['acc'], 'test': 100 * j['test']['acc']})
    return pd.DataFrame(rows, columns=['task', 'fold', 'ratio', 'val', 'test'])


def load_ft(runs, folds):
    """파인튜닝 + movement 실행의 조건 x 비율 test·val(%). fold 0·1은 예산 분할 실험(split_points.csv),
    다른 fold는 prune_grid(mov_cum, 2800 step) 결과이고 비율 0은 기준 모델 result.json이다."""
    if set(folds) <= {0, 1}:
        pts = load_points(runs)
        pts = pts[pts.index.get_level_values('fold').isin(folds)]
        return pts['test'][GRID], pts['val'][GRID]
    return load_ft_prune(runs, folds)


def load_ft_prune(runs, folds):
    """prune_grid(mov_cum, 2800 step) 결과로 만든 조건 x 비율 test·val(%). 모든 비율이 있는 조건만 남긴다."""
    d = load_prune(runs, 'mov_cum', REC_STEPS)
    d = d[d.fold.isin(folds)]
    if d.empty:
        raise SystemExit(f'fold {folds}의 prune_grid mov_cum {REC_STEPS} step 결과가 없습니다.')
    base = []
    for t, f in d[['task', 'fold']].drop_duplicates().itertuples(index=False):
        j = json.loads((runs / 'base' / t / f'fold{f}_seed0' / 'result.json').read_text(encoding='utf-8'))
        base.append({'task': t, 'fold': f, 'ratio': 0.0, 'val': 100 * j['val']['acc'], 'test': 100 * j['test']['acc']})
    d = pd.concat([d, pd.DataFrame(base)], ignore_index=True)
    test, val = (d.pivot_table(index=['task', 'fold'], columns='ratio', values=v).reindex(columns=GRID)
                 for v in ('test', 'val'))
    keep = test.notna().all(axis=1) & val.notna().all(axis=1)
    return test[keep], val[keep]


def fit(tt, rule):
    """과제별 표 tt(GRID 열 = 목표값의 fold 평균, sil, k)로 예측식을 만든다. 반환: (실루엣, 클래스 수) -> 비율 순 예측값."""
    if rule == 'FCA식':  # 원본 정확도만 실루엣으로 예측하고, 하락은 과제 평균
        a = np.polyfit(tt.sil, tt[0.0], 1)
        drop = (tt[[0.0]].to_numpy() - tt[GRID].to_numpy()).mean(axis=0)
        return lambda s, k: np.polyval(a, s) - drop
    x = MODELS[rule]
    X = np.array([x(s, k) for s, k in zip(tt.sil, tt.k)])
    coef = np.linalg.lstsq(X, tt[GRID].to_numpy(), rcond=None)[0]
    return lambda s, k: np.asarray(x(s, k)) @ coef


def climb(ok):
    """비율을 0부터 올리며 처음 실패하기 직전의 비율 (ok: GRID 순서의 통과 여부)."""
    best = 0.0
    for g, o in zip(GRID, ok):
        if not o:
            break
        best = g
    return best


def calibrate(tt, conds, V, sil, rule, thr):
    """tt의 과제들 안에서 과제 하나씩 빼고 예측했을 때 RELIABILITY 이상 지키는 가장 작은 여유."""
    pairs = []
    for u in tt.index:
        f = fit(tt.drop(index=u), rule)
        pairs += [(f(sil[c], tt.loc[u, 'k']), V.loc[c].to_numpy()) for c in conds.get(u, [])]
    for m in MARGINS:
        if pairs and np.mean([v[GRID.index(climb(p - m >= thr))] >= thr for p, v in pairs]) >= RELIABILITY:
            return m
    return MARGINS[-1]


def choose(V, Vval, sil, k, thr, rules):
    """조건마다 규칙별로 고른 비율과 여유. V, Vval: 조건 x 비율의 목표값(test, 검증). thr 이상이면 목표를 지킨다."""
    tt = V.groupby(level='task').mean()
    tt['sil'], tt['k'] = sil.groupby(level='task').mean(), k
    feas = V[0.0] >= thr  # 원본부터 목표를 지키는 조건
    task_of = V.index.get_level_values('task')
    picks, margins = {}, {}
    for t in tt.index:
        mine, others = V.index[task_of == t], tt.drop(index=t)
        rest = V[(task_of != t) & feas.to_numpy()]
        conds = {}
        for c in rest.index:
            conds.setdefault(c[0], []).append(c)
        fixed = climb([(rest[g] >= thr).mean() >= RELIABILITY for g in GRID])
        for rule in rules:
            m, f = calibrate(others, conds, V, sil, rule, thr), fit(others, rule)
            for c in mine:
                picks.setdefault(rule, {})[c] = climb(f(sil[c], tt.loc[t, 'k']) - m >= thr)
                margins.setdefault(rule, {})[c] = m
        for c in mine:
            picks.setdefault('고정', {})[c] = fixed
            picks.setdefault('탐색', {})[c] = max([g for g in GRID if Vval.loc[c, g] >= thr], default=0.0)
            picks.setdefault('사후 최적', {})[c] = max([g for g in GRID if V.loc[c, g] >= thr], default=0.0)
    return {r: pd.Series(p) for r, p in picks.items()}, {r: pd.Series(m) for r, m in margins.items()}, feas


def blind_ratio(V, ok, thr, hit):
    """과제를 보지 않고 모든 조건에 같은 비율(이웃한 두 비율을 섞는 것 포함)을 쓸 때, 달성률 hit에서의 평균 비율."""
    H = np.minimum.accumulate([(V.loc[ok, g] >= thr).mean() for g in GRID])  # 비율을 올리면 달성률은 줄기만 한다
    return float(np.interp(hit, H[::-1], np.asarray(GRID)[::-1]))


def steps(rule, g, arm):
    """학습 step: 파인튜닝 840(생략 실행은 비율 0일 때만) + 회복 2800. 탐색·사후 최적은 모든 비율을 회복한다."""
    if rule in ('탐색', '사후 최적'):
        return FT_STEPS + REC_STEPS * (len(GRID) - 1)
    if g == 0:
        return FT_STEPS
    return (FT_STEPS if arm == 'ft' else 0) + REC_STEPS


def report(title, V, Vval, sil, k, thr, size, arm, rules, detail=False):
    """한 목표·실행의 규칙별 결과를 출력하고 판정용으로 돌려준다."""
    picks, margins, feas = choose(V, Vval, sil, k, thr, rules)
    ok = feas.index[feas.to_numpy()]
    print(f'\n  {title} (원본부터 목표를 지키는 조건 {len(ok)}/{len(V)}개로 평가)')
    res = {}
    for r in ['고정', *rules, '탐색', '사후 최적']:
        s = picks[r].reindex(ok)
        hit = pd.Series([V.loc[c, g] >= thr for c, g in s.items()], index=ok)
        res[r] = {'hit': hit, 'ratio': s, 'gain': s.mean() - blind_ratio(V, ok, thr, hit.mean())}
        m = f'{margins[r].reindex(ok).mean():4.2f}' if r in margins else '   -'
        print(f'    {r:<10} 달성 {hit.mean():6.1%}  비율 {s.mean():.3f}  이득 {res[r]["gain"]:+.3f}  '
              f'크기 {np.exp(np.mean([np.log(size[g]) for g in s])):6.2f}MB(기하평균)  여유 {m}%p  '
              f'학습 {np.mean([steps(r, g, arm) for g in s]):6.0f} step')
    me = res[MAIN]
    for o in [r for r in ['고정', 'FCA식', '곡선(k)', '탐색'] if r in res]:
        x = res[o]
        b, c = int((me['hit'] & ~x['hit']).sum()), int((~me['hit'] & x['hit']).sum())
        p = binomtest(b, b + c).pvalue if b + c else float('nan')
        print(f'      {MAIN} − {o}: 달성 {me["hit"].mean() - x["hit"].mean():+6.1%} '
              f'({MAIN}만 {b}, 상대만 {c}, p={p:.3f})  비율 {fmt(*stat(me["ratio"] - x["ratio"]))}')
    if detail:
        tab = pd.DataFrame({r: res[r]['ratio'] for r in ['고정', 'FCA식', MAIN, '사후 최적'] if r in res})
        tab = tab.groupby(level='task').mean()
        tab.insert(0, '원본', V.loc[ok, 0.0].groupby(level='task').mean())
        tab.insert(0, '실루엣', sil.groupby(level='task').mean().reindex(tab.index))
        print('    과제별 고른 비율 (fold 평균):')
        print(tab.sort_values('실루엣').round(3).to_string())
    return res


def by_kind(s, y):
    """전체, 쌍, 삼중 과제의 순위 상관 문자열."""
    kind = np.asarray(y.index.str.split('_').str[0])
    return (f'전체 {rho_text(s, y)} | 쌍 {rho_text(s[kind == "pair"], y[kind == "pair"])} | '
            f'삼중 {rho_text(s[kind == "tri"], y[kind == "tri"])}')


def skip_cost(ft, skip, sil):
    print('\n[C] 파인튜닝 생략 손해: 파인튜닝 + movement − 생략 (test %p, 양수면 생략이 손해)')
    gap = ft.loc[skip.index, GRID[1:]] - skip[GRID[1:]]
    print('  비율별 평균: ' + ', '.join(f'{g:g} {gap[g].mean():+.2f}' for g in GRID[1:]))
    tg = gap.mean(axis=1).groupby(level='task').mean()
    s = sil.groupby(level='task').mean().reindex(tg.index)
    print(f'  실루엣과 순위 상관: {by_kind(s, tg)}')
    easy = s >= s.median()
    print(f'  쉬운 절반 {tg[easy].mean():+.2f}, 어려운 절반 {tg[~easy].mean():+.2f}')
    r, p = spearmanr(s, tg)
    return r < 0 and p < 0.05


def recovery_gain(runs, ft, sil, folds):
    print('\n[D] 회복 이득: 2800 step − 840 step (movement, test %p, 양수면 길게 회복할수록 이득)')
    key = ['task', 'fold', 'ratio']
    short = load_prune(runs, 'mov_cum', 840)
    short = short[short.fold.isin(folds)].rename(columns={'test': 'test_short'})[key + ['test_short']]
    same = load_prune(runs, 'mov_cum', REC_STEPS).rename(columns={'test': 'test_long'})[key + ['test_long']]
    same = same[same.fold.isin(folds)].merge(short, on=key)
    low, low_name = same[same.ratio.isin([0.4, 0.6])], '0.4·0.6 (참고, 같은 코드)'
    if low.empty:  # fold 0·1에서 0.4·0.6의 2800 step은 분할 실험 모델에만 있다
        low = (ft[[0.4, 0.6]].reset_index().melt(id_vars=['task', 'fold'], var_name='ratio', value_name='test_long')
               .astype({'ratio': float}).merge(short, on=key))
        low_name = '0.4·0.6 (참고: 2800은 분할 실험 모델)'
    s = sil.groupby(level='task').mean()
    verdict = None
    for name, d, main in [('0.8·0.9 (같은 코드)', same[same.ratio.isin([0.8, 0.9])], True), (low_name, low, False)]:
        if d.empty:
            print(f'  {name}: 결과 없음')
            continue
        d = d.assign(gain=d.test_long - d.test_short)
        tg = d.groupby('task').gain.mean()
        print(f'  {name}: 비율별 평균 ' + ', '.join(f'{r:g} {v:+.2f}' for r, v in d.groupby('ratio').gain.mean().items()))
        print(f'      실루엣과 순위 상관: {by_kind(s.reindex(tg.index), tg)}')
        if main:
            r, p = spearmanr(s.reindex(tg.index), tg)
            verdict = r < 0 and p < 0.05
    return verdict


def fold_effect(runs, ft, sim, tol, folds):
    print(f'\n[E] H2가 약해진 이유: r*(원본 대비 {tol:g}%p 안에서 자를 수 있는 비율)와 실루엣(전체 이미지)의 순위 상관')
    short = load_prune(runs, 'mov_cum', 840)
    short = short[short.ratio.isin(GRID840[1:])]
    if short.empty:
        print('  840 step 결과 없음')
        return None
    base = {}
    for t, f in short[['task', 'fold']].drop_duplicates().itertuples(index=False):
        p = runs / 'base' / t / f'fold{f}_seed0' / 'result.json'
        if p.exists():
            base[(t, f)] = 100 * json.loads(p.read_text(encoding='utf-8'))['test']['acc']
    short = short.assign(drop=[base.get((t, f), np.nan) - a for t, f, a in zip(short.task, short.fold, short.test)])
    tab = short.pivot_table(index=['task', 'fold'], columns='ratio', values='drop')

    def rho_840(grid, folds):
        sub = tab.reindex(columns=grid[1:])
        sub = sub[sub.notna().all(axis=1) & sub.index.get_level_values('fold').isin(list(folds))]
        n = sub.groupby(level='task').size()
        dt = sub.groupby(level='task').mean()[n == len(folds)]
        r = pd.Series({t: rstar(dt.loc[t].to_numpy(), tol, grid) for t in dt.index})
        return spearmanr(sim.reindex(r.index), r)[0], len(r)

    combos = [rho_840(E_GRID, c)[0] for c in itertools.combinations(range(5), 2)]
    d28 = ft.rsub(ft[0.0], axis=0).groupby(level='task').mean()
    r28 = pd.Series({t: rstar(d28.loc[t, E_GRID[1:]].to_numpy(), tol, E_GRID) for t in d28.index})
    rho28 = spearmanr(sim.reindex(r28.index), r28)[0]
    all5, n5 = rho_840(E_GRID, range(5))
    fs = '·'.join(map(str, folds))
    print(f'  참고: 840 step, fold 5개, H2의 비율 후보(0.2~0.95): {rho_840(GRID840, range(5))[0]:+.2f}')
    print(f'  840 step, fold 5개         : {all5:+.2f} (과제 {n5}개)')
    print(f'  840 step, fold 2개 조합 10개: 평균 {np.mean(combos):+.2f} (최소 {min(combos):+.2f}, 최대 {max(combos):+.2f})')
    print(f'  840 step, fold {fs:<11}: {rho_840(E_GRID, folds)[0]:+.2f}')
    print(f'  2800 step, fold {fs:<10}: {rho28:+.2f} (과제 {len(r28)}개)')
    if set(folds) != {0, 1}:
        print('  (확인 단계에서는 [E]를 판정하지 않는다)')
        return None
    return np.mean(combos) - rho28 >= 0.2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--targets', type=float, nargs='+', default=[90, 92], help='[A] 요구 test 정확도(%%). 첫 값이 주')
    ap.add_argument('--tol', type=float, default=3.0, help='[B] 원본 대비 허용 하락(%%p)')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1], help='0 1이면 탐색, 2 3 4면 확인 단계')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    runs = C.runs_dir(args.work_dir)
    explore = set(args.folds) <= {0, 1}

    ft, ft_val = load_ft(runs, args.folds)
    size = load_points(runs)['size_mb'][GRID].mean().to_dict()  # 크기는 비율로만 정해진다
    sk = load_prune(runs, 'pretrained_l2', REC_STEPS)
    sk = sk[sk.fold.isin(args.folds)]
    skip = skip_val = None
    if not sk.empty:  # 비율 0(자르지 않음)은 생략 실행에서도 파인튜닝한 원본이다
        skip, skip_val = (sk.pivot_table(index=['task', 'fold'], columns='ratio', values=v).reindex(columns=GRID)
                          for v in ('test', 'val'))
        skip[0.0], skip_val[0.0] = ft[0.0], ft_val[0.0]
        keep = skip.notna().all(axis=1) & skip_val.notna().all(axis=1)
        skip, skip_val = skip[keep], skip_val[keep]
    sil = train_silhouettes(args.work_dir, args.data_root, list(ft.index))
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    stage = '탐색' if explore else '확인'
    print(f'=== 과제별 계획 진단 (조건 {len(ft)}개, fold {"·".join(map(str, args.folds))} {stage}) ===')

    rules_a, a0 = ['FCA식', *MODELS], args.targets[0]
    print('\n[A] 배포 요구 목표: test 정확도가 요구치 이상이면서 최대한 자르기 (실행: 파인튜닝 + movement)')
    res_a = {a: report(f'요구 {a:g}%', ft, ft_val, sil, sim.k, a, size, 'ft', rules_a, detail=a == a0)
             for a in args.targets}
    res_skip = None
    if skip is not None and len(skip):
        print('\n  파인튜닝 생략 실행 (생략 결과로 규칙을 만들고 평가)')
        res_skip = report(f'요구 {a0:g}%, 생략', skip, skip_val, sil, sim.k, a0, size, 'skip', rules_a)

    print(f'\n[B] 원래 목표: 원본 대비 하락 ≤ {args.tol:g}%p (실행: 파인튜닝 + movement)')
    res_b = report(f'하락 ≤ {args.tol:g}%p', ft.sub(ft[0.0], axis=0), ft_val.sub(ft_val[0.0], axis=0), sil, sim.k,
                   -args.tol, size, 'ft', list(MODELS))

    c = skip_cost(ft, skip, sil) if skip is not None and len(skip) else None
    d = recovery_gain(runs, ft, sil, args.folds)
    e = fold_effect(runs, ft, sim.silhouette, args.tol, args.folds)

    def yes(v):
        return '보류 (결과 없음)' if v is None else ('예' if v else '아니오')

    print(f'\n[판정] ({stage} 단계)')
    for a, res in res_a.items():
        print(f'  [A] 요구 {a:g}% ({"주" if a == a0 else "보조"}): 과제별 계획 이득 {yes(res[MAIN]["gain"] >= 0.05)} '
              f'({res[MAIN]["gain"]:+.3f}), 프루닝 쪽 기여 {yes(res[MAIN]["gain"] - res["FCA식"]["gain"] >= 0.03)} '
              f'(FCA식보다 {res[MAIN]["gain"] - res["FCA식"]["gain"]:+.3f})')
    print(f'  [B] 원래 목표 회복: {yes(res_b[MAIN]["gain"] >= 0.05)} ({res_b[MAIN]["gain"]:+.3f})')
    skip_gain = None if res_skip is None else res_skip[MAIN]['gain'] - res_skip['FCA식']['gain']
    if res_skip is not None:
        print(f'  [A] 요구 {a0:g}%, 생략 실행: 과제별 계획 이득 {yes(res_skip[MAIN]["gain"] >= 0.05)} '
              f'({res_skip[MAIN]["gain"]:+.3f}), 프루닝 쪽 기여 {yes(skip_gain >= 0.03)} (FCA식보다 {skip_gain:+.3f})')
    print(f'  [C] 쉬운 과제만 생략: {yes(c)}')
    print(f'  [D] 난이도가 회복 속도를 정한다: {yes(d)}')
    if explore:
        print(f'  [E] H2가 약해진 이유: {"보류 (결과 없음)" if e is None else ("회복 길이" if e else "fold 수(잡음)")}')
    else:
        both = None if d is None else (res_a[a0][MAIN]['gain'] >= 0.05 and d)
        print(f'  확인 항목 ([A] 요구 {a0:g}% 과제별 계획 이득, [D]): '
              f'{"보류 (결과 없음)" if both is None else ("둘 다 재현됨" if both else "재현 안 됨")}')
        both2 = None if c is None or skip_gain is None else (c and skip_gain >= 0.03)
        print(f'  2차 확인 항목 ([C], 생략 실행의 FCA식 대비 기여): '
              f'{"보류 (생략 실행 결과 없음)" if both2 is None else ("둘 다 재현됨" if both2 else "재현 안 됨")}')


if __name__ == '__main__':
    main()
