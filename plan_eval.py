"""미리 정한 프루닝 계획이 실제로 통하나: 결정 방식끼리 비교한다 (이미 있는 결과를 읽고 몇 초).

목표: 원본(파인튜닝한 ResNet-18) 대비 test 하락 3%p 안에서 가장 많이 자르기 (H2의 r*와 같은 정의).
비율 후보는 0(안 자름), 0.4, 0.6, 0.7, 0.8, 0.9이고, 계획은 예측한 r*를 넘지 않는 가장 큰 후보를 고른다.
결정 방식 (규칙은 모두 평가하는 과제를 빼고 나머지 과제로 만든다):
  고정       : 다른 과제 r*의 평균 (난이도 무시)
  클래스 수  : 다른 과제로 맞춘 r* ~ 클래스 수
  실루엣     : 다른 과제로 맞춘 r* ~ 실루엣 (우리 계획)
  실루엣+k   : r* ~ 실루엣 + 클래스 수
  탐색       : 파인튜닝 -> 모든 비율을 회복 -> 검증 하락이 3%p 안인 가장 큰 비율 (보통 쓰는 방식)
  사후 최적  : 모든 비율을 회복하고 test로 고름 (상한. 실제로는 쓸 수 없다)
실루엣은 fold마다 학습 이미지만으로 다시 계산한다 (사전학습 ResNet-18 특징, 코사인).
실행은 파인튜닝 + movement + 2800 step 회복 모델이다 (split_points.csv의 fp32, 비율 0은 원본).
실루엣 계획은 파인튜닝을 생략한 실행(사전학습 가중치를 L2로 바로 잘라 2800 step 학습)도 본다.
  생략 : 모든 비율에서 생략 (prune_grid --init pretrained 결과)
  혼합 : 고른 비율이 0.8 이하면 생략, 0.9면 파인튜닝 + movement (상속 실험에서 0.9에서만 마스크가 중요했다)
학습 비용은 두 가지로 센다: step 수(파인튜닝 840 + 회복 2800 x 횟수)와, 원본 1 step을 1로 센 연산
(파인튜닝 840 + 회복 2800 x 자른 모델의 MACs 비율, summarize_inherit와 같다). 이 서버에서는 데이터 읽기가 병목이라
자른 모델도 step당 시간이 크게 줄지 않을 것으로 보고, 판정은 실제 시간에 가까운 step 수로 한다 (회복 1회 실측 시간을 함께 낸다).
주의: 원본의 검증 정확도는 최고 시점을 고른 값이라 조금 높게 나온다. 그래서 탐색은 보수적으로 고르는 쪽으로 치우친다.

판정 (결과를 보기 전에 정함):
  전제      : 실루엣 예측의 r* 평균 오차가 고정 예측보다 작고, 실루엣과 r*의 Spearman ρ>0, p<0.05
  계획      : 실루엣 계획의 목표 달성률이 80% 이상이고 고정 이상이며, 평균 비율이 고정보다 0.05 넘게 낮지 않다
  클래스 수 : 실루엣 계획의 달성률이 클래스 수 규칙 이상이고, 평균 비율이 0.05 넘게 낮지 않다
  탐색 대비 : 실루엣 계획의 달성률이 탐색보다 10%p 넘게 낮지 않고, 평균 비율이 0.05 넘게 낮지 않으며,
              학습 step이 탐색의 1/3 이하 -> '미리 정해도 된다'
  생략      : 혼합 실행의 달성률이 파인튜닝 실행보다 10%p 넘게 낮지 않다 -> '파인튜닝 생략 가능'

사용: python plan_eval.py > work/plan_summary.txt
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import silhouette_score

import common as C
from split_followup import rho_text
from summarize_budget import fmt, stat
from summarize_inherit import FT_STEPS, R18_MACS

GRID = [0.0, 0.4, 0.6, 0.7, 0.8, 0.9]
REC_STEPS = 2800
SKIP_MAX = 0.8  # 혼합 실행에서 파인튜닝을 생략하는 최대 비율
RULES = ['고정', '클래스 수', '실루엣', '실루엣+k']
HOWS = {'ft': '파인튜닝 + movement', 'skip': '생략', 'mix': '혼합'}


def rstar(drops, tol):
    """하락(비율 0.4~0.9 순)이 tol(%p)을 처음 넘는 지점을 선형 보간한 비율 (predict_budget.py와 같은 방식)."""
    x, y = np.asarray(GRID), np.r_[0.0, drops]
    for i in range(1, len(x)):
        if y[i] > tol:
            return x[i - 1] + (tol - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
    return x[-1]


def snap(r):
    """예측한 r*를 넘지 않는 가장 큰 후보."""
    return max([g for g in GRID if g <= r + 1e-9], default=0.0)


def predict(tt, task, sil):
    """task를 뺀 과제들로 규칙을 맞추고 이 과제의 r*를 예측한다 (sil: 이 과제의 실루엣)."""
    o, k = tt.drop(index=task), tt.loc[task, 'k']
    a, b = np.polyfit(o.k, o.rstar, 1), np.polyfit(o.sil, o.rstar, 1)
    c, *_ = np.linalg.lstsq(np.c_[np.ones(len(o)), o.sil, o.k], o.rstar.to_numpy(), rcond=None)
    return {'고정': o.rstar.mean(), '클래스 수': np.polyval(a, k), '실루엣': np.polyval(b, sil),
            '실루엣+k': float(c @ [1, sil, k])}


def load_skip(runs):
    """파인튜닝 생략 실행(사전학습 가중치 L2, 2800 step)의 test 정확도(%). (task, fold, ratio) 색인."""
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in runs.glob(f'prune/*/fold*_seed0/pretrained_l2_r*_rec{REC_STEPS}.json')]
    s = pd.Series({(j['task'], j['fold'], j['ratio']): 100 * j['test']['acc'] for j in rows}, dtype=float)
    if not s.empty:
        s.index.names = ['task', 'fold', 'ratio']
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--tol', type=float, default=3.0, help='원본 대비 허용 test 하락 (%%p)')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    runs, tol = C.runs_dir(args.work_dir), args.tol

    d = pd.read_csv(runs / 'split_points.csv')
    pts = d[d.bits == 'fp32'].pivot_table(index=['task', 'fold'], columns='ratio', values=['val', 'test', 'size_mb'])
    pts = pts[pts['test'][GRID].notna().all(axis=1)]
    test, val, size = pts['test'], pts['val'], pts['size_mb']
    conds = list(pts.index)
    q = {g: [json.loads(p.read_text(encoding='utf-8')) for p in runs.glob(f'qready/*/fold*_seed0/movement_r{g}.json')]
         for g in GRID[1:]}
    macs = {g: q[g][0]['macs'] / R18_MACS for g in GRID[1:]}

    def cost(g, ft=True):
        """(학습 step 수, 원본 1 step = 1로 센 연산). 비율 0은 파인튜닝만 한다."""
        if g == 0:
            return FT_STEPS, FT_STEPS
        return (FT_STEPS if ft else 0) + REC_STEPS, (FT_STEPS if ft else 0) + REC_STEPS * macs[g]

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    feats = np.load(C.meta_dir(args.work_dir) / 'feat_pretrained.npy')
    sil = {}
    for t, f in conds:
        idx, y = C.task_split(images, tasks[t], f)['train']
        sil[(t, f)] = silhouette_score(feats[idx], y, metric='cosine')
    sil = pd.Series(sil)
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    drops = test.rsub(test[0.0], axis=0).groupby(level='task').mean()  # 원본 - 비율, fold 평균
    tt = pd.DataFrame({'k': sim.k.reindex(drops.index), 'sil': sil.groupby(level=0).mean().reindex(drops.index),
                       'rstar': [rstar(drops.loc[t, GRID[1:]].to_numpy(), tol) for t in drops.index]})

    print(f'=== 미리 정한 프루닝 계획 검증 (조건 {len(conds)}개, 목표: 원본 대비 test 하락 ≤ {tol:g}%p) ===')
    print(f'\n[0] 전제: 학습 전에 r*(하락 {tol:g}%p 안에서 자를 수 있는 비율)를 예측하나 (과제 하나씩 빼고, 2800 step 회복)')
    print(f'  학습 이미지 실루엣과 전체 이미지 실루엣의 순위 상관 {spearmanr(tt.sil, sim.silhouette.reindex(tt.index))[0]:.3f}')
    task_pred = pd.DataFrame({t: predict(tt, t, tt.loc[t, 'sil']) for t in tt.index}).T
    err = {n: (task_pred[n] - tt.rstar).abs().mean() for n in RULES}
    for n in RULES:
        line = f'  {n:<6} r* 평균 오차 {err[n]:.3f}'
        if n != '고정':
            line += f'  예측과 r*의 순위 상관 {rho_text(task_pred[n], tt.rstar)}'
        print(line)
    kind = np.asarray(tt.index.str.split('_').str[0])
    print(f'  실루엣과 r*: 전체 {rho_text(tt.sil, tt.rstar)} | 쌍 {rho_text(tt.sil[kind == "pair"], tt.rstar[kind == "pair"])}'
          f' | 삼중 {rho_text(tt.sil[kind == "tri"], tt.rstar[kind == "tri"])}')

    rows = []
    search = (FT_STEPS + REC_STEPS * (len(GRID) - 1), FT_STEPS + sum(REC_STEPS * macs[g] for g in GRID[1:]))
    for t, f in conds:
        pred = predict(tt, t, sil[(t, f)])
        pick = {n: snap(pred[n]) for n in RULES}
        pick['탐색'] = max(g for g in GRID if val.loc[(t, f), 0.0] - val.loc[(t, f), g] <= tol)
        pick['사후 최적'] = max(g for g in GRID if test.loc[(t, f), 0.0] - test.loc[(t, f), g] <= tol)
        for n, g in pick.items():
            full = n in ('탐색', '사후 최적')
            rows.append({'task': t, 'fold': f, 'rule': n, 'pred': pred.get(n, np.nan), 'ratio': g,
                         'drop': test.loc[(t, f), 0.0] - test.loc[(t, f), g], 'size': size.loc[(t, f), g],
                         'runs': len(GRID) if full else 1 + (g > 0),
                         **dict(zip(['steps', 'cost'], search if full else cost(g)))})
    res = pd.DataFrame(rows)
    res['hit'] = res['drop'] <= tol
    best = res[res.rule == '사후 최적'][['task', 'fold', 'ratio']].rename(columns={'ratio': 'best'})
    res = res.merge(best, on=['task', 'fold'])
    by = {n: res[res.rule == n].set_index(['task', 'fold']).loc[conds] for n in RULES + ['탐색', '사후 최적']}

    print('\n[1] 결정 방식 비교 (실행: 파인튜닝 + movement + 2800 step 회복, 같은 조건끼리)')
    print('    달성: 하락 3%p 안인 조건 비율 | 덜 자름: 사후 최적 비율 - 고른 비율 | 학습: 실행 수, step 수, 연산(원본 1 step = 1)')
    for n, g in by.items():
        print(f'  {n:<7} 달성 {g.hit.mean():6.1%}  하락 {g["drop"].mean():5.2f}  비율 {g.ratio.mean():.3f}  '
              f'크기 {np.exp(np.log(g["size"]).mean()):6.2f}MB(기하평균)  덜 자름 {(g.best - g.ratio).mean():+.3f}  '
              f'학습 {g.runs.mean():.1f}회 {g.steps.mean():6.0f} step, 연산 {g.cost.mean():5.0f}')
    me = by['실루엣']
    dist = ', '.join(f'{r:g}:{v}' for r, v in me.ratio.value_counts().sort_index().items())
    print(f'  실루엣 계획이 고른 비율 [{dist}]')
    sec = ', '.join(f'{g:g}: {np.nanmedian([j.get("time_sec", np.nan) for j in q[g]]):.0f}s' for g in GRID[1:])
    print(f'  회복 1회 실측 시간 (qready, 평가 포함, 중앙값) {sec}')
    print('\n  짝 비교 (실루엣 계획 − 상대)')
    for n in ['고정', '클래스 수', '실루엣+k', '탐색']:
        o = by[n]
        b, c = int((me.hit & ~o.hit).sum()), int((~me.hit & o.hit).sum())
        p = binomtest(b, b + c).pvalue if b + c else float('nan')
        print(f'    대 {n:<5} 달성 {me.hit.mean() - o.hit.mean():+6.1%} (실루엣만 {b}, 상대만 {c}, p={p:.3f})  '
              f'비율 {fmt(*stat(me.ratio - o.ratio))}  test {fmt(*stat(o["drop"] - me["drop"]))}')

    skip = load_skip(runs)

    def execute(t, f, g, how):
        """(test 정확도, 학습 step, 연산). 비율 0과 'ft'는 파인튜닝 경로, 나머지는 생략 경로."""
        if g == 0 or how == 'ft' or (how == 'mix' and g > SKIP_MAX):
            return test.loc[(t, f), g], *cost(g)
        return skip.get((t, f, g), np.nan), *cost(g, ft=False)

    ex = pd.DataFrame([{'task': t, 'fold': f, 'how': how, 'ratio': g,
                        **dict(zip(['acc', 'steps', 'cost'], execute(t, f, g, how)))}
                       for (t, f), g in me.ratio.items() for how in HOWS])
    ex['drop'] = test[0.0].reindex(pd.MultiIndex.from_frame(ex[['task', 'fold']])).to_numpy() - ex.acc
    complete = ex.groupby(['task', 'fold']).acc.apply(lambda s: s.notna().all())
    ok = complete[complete].index
    print(f'\n[2] 실루엣 계획의 실행 방식 (세 방식 모두 결과가 있는 조건 {len(ok)}/{len(me)}개)')
    missing = ex[ex.acc.isna()].ratio.value_counts().sort_index()
    if not missing.empty:
        print('    생략 결과가 없는 비율: ' + ', '.join(f'{r:g} ({v}개)' for r, v in missing.items()))
    skip_ok = None
    if len(ok):
        e = {how: ex[ex.how == how].set_index(['task', 'fold']).loc[ok] for how in HOWS}
        for how, name in HOWS.items():
            print(f'  {name:<12} 달성 {(e[how]["drop"] <= tol).mean():6.1%}  하락 {e[how]["drop"].mean():5.2f}  '
                  f'학습 {e[how].steps.mean():6.0f} step, 연산 {e[how].cost.mean():5.0f}')
        for how in ('skip', 'mix'):
            print(f'  {HOWS[how]} − 파인튜닝: test {fmt(*stat(e[how].acc - e["ft"].acc))}')
        if len(ok) == len(me):
            skip_ok = (e['mix']['drop'] <= tol).mean() >= (e['ft']['drop'] <= tol).mean() - 0.10

    print('\n[3] 과제별 (fold 평균, 실루엣 계획)')
    per = pd.DataFrame({'예측r*': me.pred.groupby(level='task').mean(), '고른비율': me.ratio.groupby(level='task').mean(),
                        '최적': me.best.groupby(level='task').mean(),
                        '탐색': by['탐색'].ratio.groupby(level='task').mean(),
                        '하락': me['drop'].groupby(level='task').mean(), '달성': me.hit.groupby(level='task').mean()})
    print(tt.rename(columns={'sil': '실루엣(학습)'}).join(per).sort_values(['k', '실루엣(학습)']).round(3).to_string())

    fix, kc, se = by['고정'], by['클래스 수'], by['탐색']
    rho, p = spearmanr(tt.sil, tt.rstar)
    verdict = {
        '전제 (실루엣으로 r* 예측)': err['실루엣'] < err['고정'] and rho > 0 and p < 0.05,
        '계획 (고정 비율보다 낫다)': (me.hit.mean() >= 0.8 and me.hit.mean() >= fix.hit.mean()
                             and me.ratio.mean() >= fix.ratio.mean() - 0.05),
        '클래스 수 이상의 정보': me.hit.mean() >= kc.hit.mean() and me.ratio.mean() >= kc.ratio.mean() - 0.05,
        '미리 정해도 된다 (탐색 대비)': (me.hit.mean() >= se.hit.mean() - 0.10 and me.ratio.mean() >= se.ratio.mean() - 0.05
                               and me.steps.mean() <= se.steps.mean() / 3),
        '파인튜닝 생략 가능 (혼합 실행)': skip_ok}
    print('\n[판정]')
    for name, v in verdict.items():
        print(f'  {name}: {"보류 (생략 결과가 모자람)" if v is None else ("예" if v else "아니오")}')


if __name__ == '__main__':
    main()
