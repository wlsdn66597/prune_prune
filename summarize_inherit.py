"""파인튜닝을 생략하고 잘라도 되나: 자른 작은 모델이 무엇을 물려받아야 하는지 비교한다 (prune_grid --init 결과).

  ft_mov  : 파인튜닝(840 step) -> movement로 자르기 -> 회복   (지금까지의 기본. 기준 x 예산 결과)
  ft_l2   : 파인튜닝 -> L2로 자르기 -> 회복                   (기준 x 예산 결과)
  pre_l2  : 사전학습 가중치를 바로 L2로 자르기 -> 학습         (파인튜닝 생략)
  pre_mov : 사전학습 가중치에 movement 마스크(파인튜닝에서 얻음)를 적용 -> 학습
            (마스크 효과와 물려받은 가중치 효과를 나눠 보는 분석용. 마스크를 얻으려면 파인튜닝이 필요하다)
학습 연산량은 원본 ResNet-18 학습 1 step을 1로 센다: 파인튜닝 840 + 회복 step x (자른 모델 MACs 비율).
기성 소형 모델(train_mobile.py)과 지연시간(measure_latency.py) 결과가 있으면 같이 보여준다.

판정 규칙 (결과를 보기 전에 정함):
  비율 0.8, 2800 step에서 pre_l2 - ft_l2 평균이 -0.5%p 이상이면 '파인튜닝 생략 가능'
  (같은 정확도를 약 7배 적은 학습 연산량으로 얻는다는 뜻)

사용: python summarize_inherit.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import common as C
from summarize_budget import fmt, stat

ARMS = {'mov_cum': 'ft_mov', 'l2': 'ft_l2', 'pretrained_l2': 'pre_l2', 'pretrained_mov_cum': 'pre_mov'}
USES_FT = {'ft_mov': True, 'ft_l2': True, 'pre_l2': False, 'pre_mov': True}
MOBILE = {'mnv3s': 'mnv3_small', 'mnv2': 'mnv2'}  # train_mobile 이름 -> 지연시간 모델 이름
FT_STEPS, R18_MACS, R18_PARAMS = 840, 1.8217e9, 11.18e6


def load_latency(work):
    """{'기기/스레드': {모델: ms}}"""
    lat = {}
    for p in sorted((Path(work) / 'latency_models').glob('latency_*.json')):
        for name, d in json.loads(p.read_text(encoding='utf-8'))['ms'].items():
            for t, ms in d.items():
                lat.setdefault(f'{p.stem[len("latency_"):]}/{t}thr', {})[name] = ms
    return lat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.8, 0.9])
    ap.add_argument('--budgets', type=int, nargs='+', default=[840, 2800])
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    runs, L = C.runs_dir(args.work_dir), max(args.budgets)

    rows = []
    for tag, arm in ARMS.items():
        for r in args.ratios:
            for s in args.budgets:
                for p in runs.glob(f'prune/*/fold*_seed0/{tag}_r{r}_rec{s}.json'):
                    j = json.loads(p.read_text(encoding='utf-8'))
                    if j['fold'] in args.folds:
                        rows.append({'task': j['task'], 'fold': j['fold'], 'ratio': r, 'arm': arm, 'steps': s,
                                     'acc': 100 * j['test']['acc'], 'macs_ratio': j['macs_ratio'],
                                     'params': j['params']})
    if not rows:
        raise SystemExit('결과가 없습니다.')
    df = pd.DataFrame(rows)
    conds = df[['task', 'fold']].drop_duplicates()
    base = pd.DataFrame([{'task': t, 'fold': f, 'acc': 100 * json.loads(
        (runs / 'base' / t / f'fold{f}_seed0' / 'result.json').read_text(encoding='utf-8'))['test']['acc']}
        for t, f in conds.itertuples(index=False)])
    mob = [json.loads(p.read_text(encoding='utf-8')) for p in runs.glob('mobile/*/fold*_seed0/*.json')]
    mob = pd.DataFrame([{'task': j['task'], 'fold': j['fold'], 'arch': j['arch'], 'val': j['val']['acc'],
                         'val_nll': j['val']['nll'], 'acc': 100 * j['test']['acc'], 'macs': j['macs'],
                         'params': j['params']} for j in mob if j['fold'] in args.folds and j['arch'] in MOBILE])
    if not mob.empty:  # (과제, fold, 모델)마다 학습률을 검증 정확도로 고른다
        mob = mob.sort_values(['val', 'val_nll'], ascending=[False, True]).groupby(
            ['task', 'fold', 'arch'], as_index=False).first()
    lat = load_latency(args.work_dir)
    print(f'=== 파인튜닝 생략·가중치 상속 (test 정확도 %, fold {args.folds}, 조건 {len(conds)}개) ===')
    print('학습 연산량 = 원본 ResNet-18 학습 1 step을 1로 센 값 (파인튜닝 840 + 회복 step x MACs 비율)')

    print('\n[0] 기준점: 원본과 기성 소형 모델 (자르지 않음, 840 step 파인튜닝)')
    ref = [{'model': 'resnet18', 'acc': base.acc.mean(), 'n': len(base), 'params_M': R18_PARAMS / 1e6,
            'train_cost': FT_STEPS, **{k: v.get('resnet18') for k, v in lat.items()}}]
    for arch, g in (mob.groupby('arch') if not mob.empty else []):
        ref.append({'model': arch, 'acc': g.acc.mean(), 'n': len(g), 'params_M': g.params.mean() / 1e6,
                    'train_cost': FT_STEPS * g.macs.mean() / R18_MACS, **{k: v.get(MOBILE[arch]) for k, v in lat.items()}})
    print(pd.DataFrame(ref).set_index('model').round(2).to_string())

    w = df.pivot_table(index=['task', 'fold', 'ratio'], columns=['arm', 'steps'], values='acc')
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    verdict = None
    for r in args.ratios:
        d = df[df.ratio == r]
        tab = d.groupby(['arm', 'steps']).agg(acc=('acc', 'mean'), n=('acc', 'size'), macs_ratio=('macs_ratio', 'mean'),
                                              params_M=('params', 'mean')).reset_index()
        tab['params_M'] /= 1e6
        tab['train_cost'] = [FT_STEPS * USES_FT[a] + s * m for a, s, m in zip(tab.arm, tab.steps, tab.macs_ratio)]
        for k, v in lat.items():
            tab[k] = v.get(f'resnet18_u{r}')
        print(f'\n==================== 비율 {r} ====================')
        print('[1] 방법별 평균')
        print(tab.drop(columns='macs_ratio').set_index(['arm', 'steps']).round(2).to_string())

        x = w.xs(r, level='ratio')
        print('\n[2] 짝지은 차이 (%p, 같은 과제·fold): 평균 [승/무/패] Wilcoxon p')
        pairs = [(('pre_l2', s), ('ft_l2', s), f'파인튜닝 생략 효과 (L2, {s} step)') for s in args.budgets]
        pairs += [(('pre_l2', L), ('ft_mov', L), '파인튜닝 생략 vs 지금 방식(ft_mov)'),
                  (('pre_mov', L), ('ft_mov', L), '같은 movement 마스크: 사전학습 - 파인튜닝 가중치'),
                  (('pre_mov', L), ('pre_l2', L), '같은 사전학습 가중치: movement - L2 마스크'),
                  (('ft_mov', L), ('ft_l2', L), '같은 파인튜닝 가중치: movement - L2 마스크')]
        for a, b, label in pairs:
            if a in x.columns and b in x.columns:
                dd = (x[a] - x[b]).dropna()
                print(f'  {fmt(*stat(dd))}  n={len(dd):>3}  {label}')
        if not mob.empty:
            for arch, g in mob.groupby('arch'):
                acc = g.set_index(['task', 'fold']).acc
                for arm in ('ft_mov', 'pre_l2'):
                    if (arm, L) in x.columns:
                        dd = (acc - x[(arm, L)]).dropna()
                        print(f'  {fmt(*stat(dd))}  n={len(dd):>3}  {arch} - {arm} ({L} step)')

        if ('pre_l2', L) in x.columns and ('ft_l2', L) in x.columns:
            dd = (x[('pre_l2', L)] - x[('ft_l2', L)]).dropna()
            g = dd.groupby(level='task').mean()
            s, kind = sim.loc[g.index, 'silhouette'], g.index.str.split('_').str[0]
            print(f'\n[3] 파인튜닝 생략 손해(pre_l2 - ft_l2, {L} step)와 과제 난이도의 순위 상관 (과제 평균)')
            for name, m in (('전체', kind == kind), ('쌍', kind == 'pair'), ('삼중', kind == 'tri')):
                if m.sum() >= 5:
                    rho, p = spearmanr(s[m], g[m])
                    print(f'  {name}: 실루엣과 {rho:+.2f} (p={p:.3f}, n={m.sum()})')
            if r == 0.8:
                verdict = stat(dd)

    if verdict is not None:
        m, wtl, p = verdict
        print(f'\n[판정] 비율 0.8, {L} step: pre_l2 - ft_l2 = {m:+.2f}%p [{wtl}] p={p:.3f}'
              f' => {"파인튜닝 생략 가능" if m >= -0.5 else "파인튜닝 생략하면 손해"}')


if __name__ == '__main__':
    main()
