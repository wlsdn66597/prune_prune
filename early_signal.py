"""0단계: 파인튜닝 도중 '언제' 잘라도 되는지 기존 기준 모델 기록으로 확인한다 (학습 없음, CPU).

기준 모델(runs/base)의 movement.pt에는 56 step마다 누적 movement가 저장돼 있다. 시점마다 그 movement로
균일하게 자를 때 남는 채널을 구해, 실제로 쓴 시점(검증 최고, best_step)에 남는 채널과 비교한다.

  일치율      : 두 마스크에서 남는 채널이 겹치는 비율 (단위 12개, 남는 채널 수로 가중)
  정착 시점   : 실제로 쓴 마스크와의 일치율이 처음 tau 이상이 되는 step (사후에만 알 수 있음)
  온라인 신호 : 바로 앞 시점 마스크와의 일치율이 처음 tau 이상이 되는 step
                (학습 중에 알 수 있음. Early-Bird, ICLR'20의 마스크 거리 방식)
  조기 종료   : best_step (검증 최고), near_best_step (최고와 1%p 안으로 처음 들어온 시점)
  K=0 기준    : 사전학습 가중치 L2, 파인튜닝 가중치 L2로 자를 때 실제 마스크와의 일치율

1단계(일찍 자르기 실험)로 갈지 정하는 기준 (결과를 보기 전에 정함, 비율 0.9):
  (a) 온라인 신호(tau=0.9)가 near_best_step보다 평균 112 step(평가 2번) 이상 빠르고
  (b) 신호 시점의 마스크가 실제 마스크와 평균 85% 이상 일치
  둘 다 만족하면 진행. (a)만 실패하면 조기 종료로 충분하다는 뜻, (b)만 실패하면 초반 movement가 잡음이라는 뜻
  (분류 헤드가 무작위인 초반 문제, LP-FT로 보완할 여지)

사용: python early_signal.py > work/early_signal.txt
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch_pruning as tp
from scipy.stats import spearmanr

import common as C
import prune_common as P
from trp_joint import config_of
from trp_search import UNIT_ROOTS, Ranker, build_dg

GROUPS = {'pair': 'pair', 'tri': 'triple', 'all6': 'multi', 'mat5': 'multi'}


def uniform_cfg(ratio):
    """균일 프루닝에서 단위별로 남는 채널 수. 구조는 가중치와 무관하다."""
    m = C.build_model(6, pretrained=False)
    P.prune(m, P.make_importance('l2', m, None), ratio)
    return config_of(m)


def unit_groups(model):
    """단위별 전체 채널 그룹 (점수 계산용, 자르지 않는다)."""
    dg, mods = build_dg(model), dict(model.named_modules())
    return {r: dg.get_pruning_group(mods[r], tp.prune_conv_out_channels, idxs=list(range(mods[r].out_channels)))
            for r in UNIT_ROOTS}


def kept(model, groups, ranker, cfg):
    """단위별로 남는 채널 인덱스. trp_joint.materialize와 같은 규칙으로 점수가 낮은 채널부터 지운다."""
    out, mods = {}, dict(model.named_modules())
    for r, g in groups.items():
        n = mods[r].out_channels
        drop = set(torch.argsort(ranker.scores(model, g))[:n - cfg[r]].tolist())
        out[r] = set(range(n)) - drop
    return out


def agree(a, b):
    return sum(len(a[r] & b[r]) for r in a) / sum(len(a[r]) for r in a)


def rho(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 5:
        return '-'
    r, p = spearmanr(x[ok], y[ok])
    return f'{r:+.2f} (p={p:.3f}, n={ok.sum()})'


def prune_gap(runs, task, fold, ratio, steps):
    """기준 x 예산 결과에서 mov_cum - l2 test 정확도(%p). 없으면 nan."""
    d = runs / 'prune' / task / f'fold{fold}_seed0'
    try:
        acc = {c: json.loads((d / f'{c}_r{ratio}_rec{steps}.json').read_text(encoding='utf-8'))['test']['acc']
               for c in ('mov_cum', 'l2')}
    except FileNotFoundError:
        return np.nan
    return 100 * (acc['mov_cum'] - acc['l2'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.8, 0.9])
    ap.add_argument('--tau', type=float, nargs='+', default=[0.8, 0.9])
    args = ap.parse_args()
    pd.set_option('display.width', 220)

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    cfgs = {r: uniform_cfg(r) for r in args.ratios}
    runs = C.runs_dir(args.work_dir)
    pre = {}  # 클래스 수별 사전학습 모델 (K=0 기준)
    rows, curves = [], []
    dirs = [d for d in sorted(runs.glob('base/*/fold*_seed0')) if (d / 'result.json').exists() and d.parent.name in tasks]
    for n, d in enumerate(dirs, 1):
        task, fold = d.parent.name, int(d.name.split('_')[0][4:])
        res = json.loads((d / 'result.json').read_text(encoding='utf-8'))
        k = len(tasks[task]['classes'])
        base, move = P.load_base(d, k, 'cpu'), P.load_movement(d)
        groups = unit_groups(base)
        if k not in pre:
            pm = C.build_model(k, pretrained=True).eval()
            pre[k] = (pm, unit_groups(pm))
        pm, pgroups = pre[k]
        steps, b, curve = move['steps'], move['best_i'], res['val_acc_curve']
        near = next(s for s, a in zip(steps, curve) if a >= max(curve) - 0.01)
        for r in args.ratios:
            cfg = cfgs[r]
            masks = [kept(base, groups, Ranker.build('mov_cum', base, dict(move, best_i=j)), cfg)
                     for j in range(len(steps))]
            to_ref = [agree(m, masks[b]) for m in masks]
            to_prev = [np.nan] + [agree(masks[j], masks[j - 1]) for j in range(1, len(masks))]
            row = {'task': task, 'fold': fold, 'k': k, 'ratio': r, 'silhouette': sim.loc[task, 'silhouette'],
                   'n_train': res['n_train'], 'best_step': steps[b], 'near_best_step': near,
                   'l2_pre': agree(kept(pm, pgroups, Ranker.build('l2', pm, None), cfg), masks[b]),
                   'l2_ft': agree(kept(base, groups, Ranker.build('l2', base, None), cfg), masks[b]),
                   'gap840': prune_gap(runs, task, fold, r, 840), 'gap2800': prune_gap(runs, task, fold, r, 2800)}
            for K in (56, 168, 280):
                row[f'agree@{K}'] = to_ref[steps.index(K)]
            for t in args.tau:
                row[f'settle@{t}'] = next(s for s, a in zip(steps, to_ref) if a >= t)
                j = next((j for j, a in enumerate(to_prev) if a >= t), None)
                row[f'online@{t}'] = np.nan if j is None else steps[j]
                row[f'online_agree@{t}'] = np.nan if j is None else to_ref[j]
            rows.append(row)
            curves += [{'task': task, 'ratio': r, 'step': s, 'to_ref': a, 'to_prev': p}
                       for s, a, p in zip(steps, to_ref, to_prev)]
        print(f'[{n}/{len(dirs)}] {task} fold{fold}', flush=True)

    df = pd.DataFrame(rows)
    df['group'] = df.task.str.split('_').str[0].map(GROUPS)
    cv = pd.DataFrame(curves)
    cv['group'] = cv.task.str.split('_').str[0].map(GROUPS)
    df.to_csv(runs / 'early_signal.csv', index=False)
    print(f'\n기준 모델 {df[["task", "fold"]].drop_duplicates().shape[0]}개, 결과 표: {runs / "early_signal.csv"}')

    tau = max(args.tau)
    for r in args.ratios:
        d, c = df[df.ratio == r], cv[cv.ratio == r]
        print(f'\n==================== 비율 {r} ====================')
        tab = c.groupby(['step', 'group']).to_ref.mean().unstack()
        tab['all'] = c.groupby('step').to_ref.mean()
        print('\n[1] 시점별 마스크 일치율 (실제로 쓴 best_step 마스크 대비)')
        print(tab.round(3).to_string())

        cols = ['best_step', 'near_best_step'] + [f'settle@{t}' for t in args.tau] + [f'online@{t}' for t in args.tau]
        cols += [f'online_agree@{t}' for t in args.tau] + ['agree@56', 'agree@168', 'agree@280', 'l2_pre', 'l2_ft']
        summ = d.groupby('group')[cols].mean()
        summ.loc['all'] = d[cols].mean()
        print('\n[2] 시점 요약 (평균, step) / 일치율 (평균)')
        print(summ.round(3).T.to_string())
        lead = d['near_best_step'] - d[f'online@{tau}']
        print(f'\n  온라인 신호(tau={tau})가 near_best_step보다 빠른 비율: {(lead > 0).mean():.2f}, '
              f'평균 {lead.mean():+.0f} step 빠름, 신호가 안 뜬 경우 {d[f"online@{tau}"].isna().mean():.2f}')

        t = d.groupby('task')[['silhouette', 'k', 'best_step', f'settle@{tau}', f'online@{tau}', 'agree@56']].mean()
        g = t.index.str.split('_').str[0]
        print('\n[3] 과제 난이도와의 순위 상관 (과제 평균, Spearman). 클래스 수를 고정한 쌍·삼중 안에서도 본다')
        for col in ('best_step', f'settle@{tau}', f'online@{tau}', 'agree@56'):
            print(f'  {col:<14} vs 실루엣: 전체 {rho(t.silhouette.values, t[col].values)}'
                  f' | 쌍 {rho(t.silhouette[g == "pair"].values, t[col][g == "pair"].values)}'
                  f' | 삼중 {rho(t.silhouette[g == "tri"].values, t[col][g == "tri"].values)}')
        print(f'  {"settle":<14} vs 클래스 수: {rho(t.k.values.astype(float), t[f"settle@{tau}"].values)}')

        print('\n[4] movement와 L2 마스크가 다를수록 movement의 정확도 이점이 큰가 (결과가 있는 조건별 Spearman)')
        dis = 1 - d['l2_ft'].values
        print(f'  (1 - L2 일치율) vs mov-L2 격차: 840 step {rho(dis, d.gap840.values)} | 2800 step {rho(dis, d.gap2800.values)}')

        if r == 0.9:
            ok_a = lead.mean() >= 112
            ok_b = d[f'online_agree@{tau}'].mean() >= 0.85
            print(f'\n[판정] (a) 신호가 조기 종료보다 112 step 이상 빠름: {"충족" if ok_a else "미충족"}'
                  f' / (b) 신호 시점 마스크 일치율 85% 이상: {"충족" if ok_b else "미충족"}'
                  f' => {"1단계 진행" if ok_a and ok_b else "1단계 보류"}')


if __name__ == '__main__':
    main()
