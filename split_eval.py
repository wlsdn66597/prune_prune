"""같은 크기 예산에서 프루닝과 양자화를 어떻게 나눌까. 그 답이 few-class 과제 난이도에 따라 달라지나.

과제·fold마다 movement로 균일하게 자른 모델을 여러 비트로 양자화해 크기와 정확도를 잰다.
  비율 0          : 자르지 않은 기준 모델 (runs/base, 파인튜닝 840 step)
  비율 0.4 ~ 0.9  : movement로 자르고 2800 step 회복한 모델 (runs/qready/.../movement_r{비율}.pt)
양자화는 측정 도구로만 쓴다. 강건성 확인에서 가장 현실적이었던 방식이다:
  conv 가중치만, 출력 채널별 대칭, 필터마다 튀는 값을 잘라낸 눈금(mse), 양자화 뒤 BN 재보정.
  비트: fp32, W8, mix6, mix5, W4 (mix는 4비트 오차가 큰 필터부터 8비트로 올려 평균을 맞춘다)
크기 = conv 가중치 비트 합 + 나머지 파라미터(BN, fc) 32비트. 1MB = 1e6 바이트.
예산마다, 크기가 예산 이하인 (비율, 비트) 조합 중 검증 정확도가 가장 높은 것을 고르고 그 test 정확도를 쓴다.

미리 정한 가설과 판정 (결과를 보기 전에 정함):
  H5     : 예산 1·0.5·0.25MB에서 과제마다 고른 프루닝 비율이 실루엣과 양의 순위 상관을 가진다
           (쉬운 과제일수록 더 자르고 비트를 높게). 세 예산 중 두 곳 이상에서 Spearman ρ>0, p<0.05이면 지지
  실용성 : 위 세 예산 평균으로, 실루엣만 보고 비율을 정하는 규칙(다른 과제로 정한 쉬움·어려움별 최적 비율)이
           모든 과제에 같은 비율을 쓸 때보다 test 정확도가 0.5%p 이상 높으면 '난이도에 맞춘 선택이 이득'

사용: python split_eval.py > work/split_summary.txt
      python split_eval.py --tasks pair_gl-pl --folds 0 --ratios 0 0.8   # 빠른 시험
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import common as C
import prune_common as P
from qready_eval import plans_for, quantized
from summarize_budget import fmt, stat

BUDGETS = [16, 8, 4, 2, 1, 0.5, 0.25, 0.1]
H5_BUDGETS = [1, 0.5, 0.25]


def size_mb(model, plan):
    """conv 가중치는 필터별 비트, 나머지 파라미터는 32비트로 센 크기 (plan이 없으면 전부 32비트)."""
    convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    conv_params = sum(m.weight.numel() for m in convs.values())
    bits = 32 * (sum(p.numel() for p in model.parameters()) - conv_params)
    if plan is None:
        bits += 32 * conv_params
    else:
        bits += sum(int((plan[n] * m.weight[0].numel()).sum()) for n, m in convs.items())
    return bits / 8 / 1e6


def evaluate(model, cache, split):
    return {k: 100 * C.metrics(C.predict_logits(model, cache, split[k][0]), split[k][1])['acc'] for k in ('val', 'test')}


def load_model(args, task, fold, ratio, k):
    runs = C.runs_dir(args.work_dir)
    if ratio == 0:
        return P.load_base(runs / 'base' / task / f'fold{fold}_seed0', k, args.device)
    path = runs / args.model_name / task / f'fold{fold}_seed0' / f'movement_r{ratio}.pt'
    return torch.load(path, map_location=args.device, weights_only=False).eval() if path.exists() else None


def pick(g, budget, ratio=None):
    """예산 안에서 검증 정확도가 가장 높은 조합 (동률이면 작은 쪽)."""
    m = g.size_mb <= budget
    if ratio is not None:
        m &= g.ratio == ratio
    feas = g[m]
    return None if feas.empty else feas.sort_values(['val', 'size_mb'], ascending=[False, True]).iloc[0]


def rho_text(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return '-', None
    r, p = spearmanr(x, y)
    return f'{r:+.2f} (p={p:.3f}, n={len(x)})', (r, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--model-name', default='qready', help='자른 모델 폴더 이름')
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--ratios', type=float, nargs='+', default=[0, 0.4, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--from-csv', action='store_true', help='평가는 건너뛰고 저장된 split_points.csv로 분석만 다시 한다')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    runs = C.runs_dir(args.work_dir)
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    if args.from_csv:
        analyze(pd.read_csv(runs / 'split_points.csv'), runs, sim)
        return
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    cache = C.EvalCache(args.work_dir, args.device)
    names = list(tasks) if args.tasks == 'all' else args.tasks.split(',')

    rows = []
    for t in names:
        for f in args.folds:
            split = C.task_split(images, tasks[t], f)
            for r in args.ratios:
                model = load_model(args, t, f, r, len(tasks[t]['classes']))
                if model is None:
                    print(f'모델 없음, 건너뜀: {t} fold{f} 비율 {r}', flush=True)
                    continue
                base_row = {'task': t, 'fold': f, 'ratio': r}
                rows.append({**base_row, 'bits': 'fp32', 'size_mb': size_mb(model, None), **evaluate(model, cache, split)})
                plans = plans_for(model, 'mse')
                plans['W8'] = {n: torch.full((m.weight.shape[0],), 8) for n, m in model.named_modules()
                               if isinstance(m, nn.Conv2d)}
                for b in ('W8', 'mix6', 'mix5', 'W4'):
                    qm = quantized(model, plans[b], 'mse')
                    P.recalibrate_bn(qm, cache, split['train'][0])
                    rows.append({**base_row, 'bits': b, 'size_mb': size_mb(model, plans[b]), **evaluate(qm, cache, split)})
                    del qm
                del model
            print(f'{t} fold{f} 완료', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(runs / 'split_points.csv', index=False)
    analyze(df, runs, sim)


def analyze(df, runs, sim):
    print(f'\n=== 프루닝·양자화 예산 분할 (조건 {df[["task", "fold"]].drop_duplicates().shape[0]}개, test 정확도 %) ===')
    print('양자화: conv만, 출력 채널별 대칭, mse 눈금, 양자화 뒤 BN 재보정 (측정 도구)')
    order = ['fp32', 'W8', 'mix6', 'mix5', 'W4']
    print('\n[0] 비율 x 비트별 평균 크기(MB)')
    print(df.pivot_table(index='ratio', columns='bits', values='size_mb')[order].round(3).to_string())
    print('\n    비율 x 비트별 평균 test 정확도')
    print(df.pivot_table(index='ratio', columns='bits', values='test')[order].round(2).to_string())

    by_ratio, choice = [], []
    for (t, f), g in df.groupby(['task', 'fold']):
        for B in BUDGETS:
            for r in sorted(g.ratio.unique()):
                b = pick(g, B, r)
                if b is not None:
                    by_ratio.append({'task': t, 'fold': f, 'budget': B, 'ratio': r, 'bits': b.bits, 'val': b.val,
                                     'test': b.test})
            b = pick(g, B)
            if b is not None:
                choice.append({'task': t, 'fold': f, 'budget': B, 'ratio': b.ratio, 'bits': b.bits, 'val': b.val,
                               'test': b.test})
    br, ch = pd.DataFrame(by_ratio), pd.DataFrame(choice)
    if br.empty:
        raise SystemExit('예산 안에 들어오는 조합이 없습니다.')
    ch.to_csv(runs / 'split_choice.csv', index=False)

    print('\n[1] 예산별, 비율별 평균 test 정확도 (각 비율에서 예산 안의 가장 좋은 비트를 검증으로 고름, 빈칸은 예산 초과)')
    print(br.pivot_table(index='budget', columns='ratio', values='test').sort_index(ascending=False).round(2).to_string())

    print('\n[2] 예산별 선택: 고른 비율의 분포, 고정 비율 대 과제별 선택 대 실루엣 규칙 (test %, 같은 과제·fold끼리 짝)')
    gaps, h5 = [], {}
    for B in BUDGETS:
        c, bB = ch[ch.budget == B], br[br.budget == B]
        if c.empty:
            continue
        dist = ', '.join(f'{r:g}:{v:.0%}' for r, v in c.ratio.value_counts(normalize=True).sort_index().items())
        r_fix = bB.groupby('ratio').val.mean().idxmax()
        fixed = bB[bB.ratio == r_fix].set_index(['task', 'fold']).test
        own = c.set_index(['task', 'fold']).test
        line = f'  {B:>5g}MB  비율 분포 [{dist}]  고정 {r_fix:g}: {fixed.mean():.2f}  과제별 선택: {own.mean():.2f}'
        ts = sorted(bB.task.unique())
        if len(ts) >= 4:  # 실루엣 규칙: 다른 과제들로 쉬움·어려움 절반별 최적 비율을 정해 적용 (학습 없이 실루엣만 필요)
            rule = {}
            for t in ts:
                others = [u for u in ts if u != t]
                med = np.median([sim.loc[u, 'silhouette'] for u in others])
                easy = sim.loc[t, 'silhouette'] >= med
                same = [u for u in others if (sim.loc[u, 'silhouette'] >= med) == easy]
                r_t = bB[bB.task.isin(same)].groupby('ratio').val.mean().idxmax()
                for _, x in bB[(bB.task == t) & (bB.ratio == r_t)].iterrows():
                    rule[(t, x.fold)] = x.test
            rule = pd.Series(rule)
            d = (rule - fixed.reindex(rule.index)).dropna()
            line += f'  실루엣 규칙: {rule.mean():.2f} (고정 대비 {fmt(*stat(d))})'
            if B in H5_BUDGETS:
                gaps.append(d.mean())
        print(line)
        if B in H5_BUDGETS:
            per_task = c.groupby('task').ratio.mean()
            h5[B] = per_task
    print('\n[3] 과제마다 고른 프루닝 비율(fold 평균)과 실루엣의 순위 상관 (H5: 쉬운 과제일수록 더 자르나)')
    support = 0
    for B, per_task in h5.items():
        s = sim.loc[per_task.index, 'silhouette']
        kind = per_task.index.str.split('_').str[0]
        txt, res = rho_text(s, per_task)
        pairs, _ = rho_text(s[kind == 'pair'], per_task[kind == 'pair'])
        tris, _ = rho_text(s[kind == 'tri'], per_task[kind == 'tri'])
        print(f'  {B:>5g}MB  전체 {txt} | 쌍 {pairs} | 삼중 {tris}')
        if res is not None and res[0] > 0 and res[1] < 0.05:
            support += 1
    if h5:
        print('\n[판정]')
        print(f'  H5: 양의 유의한 상관이 나온 예산 {support}/{len(h5)}개 -> {"지지" if support >= 2 else "지지 안 됨"}')
        if gaps:
            g = float(np.mean(gaps))
            print(f'  실용성: 실루엣 규칙 - 고정 비율 = 평균 {g:+.2f}%p (1·0.5·0.25MB) -> '
                  f'{"난이도에 맞춘 선택이 이득" if g >= 0.5 else "고정 비율로 충분"}')


if __name__ == '__main__':
    main()
