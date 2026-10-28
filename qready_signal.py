"""0단계 (양자화 대비 프루닝): 남길 채널을 고를 때 양자화하기 쉬운 쪽을 택할 여유가 있는지 본다 (학습 없음, CPU).

같은 너비(균일 비율)에서 어떤 채널을 남길지만 바꿔 비교한다. 남은 네트워크가 채널 단위 4비트 양자화에 얼마나
불리한지, 그리고 과제 중요도(movement)를 얼마나 잃는지를 잰다.
양자화는 팀 설계(출력 채널마다 비트를 정하는 가중치 양자화)를 따라 대칭, 출력 채널별, 최댓값 눈금, 반올림으로 가정한다.
conv만 보고 BN은 합치지 않는다. 팀원 방식이 정해지면 qerr()만 바꾸면 된다.

  movement : movement 점수가 높은 채널을 남긴다 (지금 방식)
  l2       : 가중치 크기가 큰 채널을 남긴다
  qr_own@λ : 중요도 z점수 - λ x (그 채널 필터가 가진 4비트 오차 몫)의 z점수가 높은 채널을 남긴다
  qr_full@λ: 위의 오차 몫에 '그 채널을 지우면 다음 층 필터들의 4비트 오차가 줄어드는 몫'을 더한다
             (출력 채널별 눈금은 필터의 최댓값으로 정해지므로, 최댓값이 든 입력 조각을 지우면 눈금이 촘촘해진다)
지표 (자른 직후 가중치 기준):
  q4      : 남은 conv 필터들의 4비트 상대 오차 평균 (파라미터 가중). 낮을수록 양자화가 적은 비트로 처리할 수 있다
  crest   : 필터의 최댓값 / RMS 평균 (튀는 값이 얼마나 큰지)
  imp_pct : 남긴 채널의 movement 순위 백분위 평균 (0.5는 무작위 수준, movement 마스크가 최대)
  ov_mov, ov_l2 : movement, l2 마스크와 남긴 채널이 겹치는 비율

1단계로 갈지 정하는 기준 (결과를 보기 전에 정함, 비율 0.8):
  어떤 λ에서 qr_full의 q4가 movement와 l2 마스크 둘 다보다 평균 10% 이상 낮고, imp_pct가 l2 마스크 이상이면 진행.
  (비율 0.8에서는 l2와 movement의 회복 후 정확도가 같았다. 그래서 l2만큼 중요도를 지키면 정확도 손해가 없다고 본다)

사용: python qready_signal.py > work/qready_signal.txt
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import common as C
import prune_common as P
from early_signal import agree, uniform_cfg, unit_groups
from trp_search import Ranker

LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def qerr(W, bits, scale=None):
    """출력 채널별 대칭 양자화의 오차 제곱을 [출력, 입력] 조각별로 낸다. scale이 없으면 필터 최댓값으로 정한다."""
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        scale = W.abs().amax(dim=(1, 2, 3)) / qmax
    s = scale.clamp_min(1e-12).view(-1, 1, 1, 1)
    q = (W / s).round().clamp(-qmax, qmax) * s
    return ((W - q) ** 2).sum(dim=(2, 3))


def own_mass(W, bits):
    """필터(출력 채널)마다 상대 오차 x 파라미터 수. 그 채널을 지우면 사라지는 오차 몫이다."""
    return W[0].numel() * qerr(W, bits).sum(1) / (W ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-12)


def removal_gain(W, bits):
    """입력 조각 c를 지울 때 이 conv의 필터들의 상대 오차가 줄어드는 양의 합 (필터 파라미터 수 가중) [입력]."""
    qmax = 2 ** (bits - 1) - 1
    e, n = qerr(W, bits), (W ** 2).sum(dim=(2, 3))
    E, N = e.sum(1, keepdim=True), n.sum(1, keepdim=True)
    rel_wo = (E - e) / (N - n).clamp_min(1e-12)  # 눈금이 그대로일 때
    top = W.abs().amax(dim=(2, 3)).topk(2, dim=1)
    a, second = top.indices[:, 0], top.values[:, 1]  # 최댓값이 든 조각과, 그 조각을 빼면 새 최댓값
    e2 = qerr(W, bits, second / qmax)
    o = torch.arange(W.shape[0])
    rel_wo[o, a] = (e2.sum(1) - e2[o, a]) / (N[:, 0] - n[o, a]).clamp_min(1e-12)
    return W[0].numel() * (E / N.clamp_min(1e-12) - rel_wo).sum(0)


def channel_costs(groups, convs, bits):
    """단위별로 채널을 지울 때 줄어드는 4비트 오차 몫: 그 채널의 필터(own), 다음 층 필터들의 눈금 좁히기(tight)."""
    roles, own, tight, gain_cache = {'out': {}, 'in': {}}, {}, {}, {}
    name_of = {id(w): n for n, w in convs.items()}
    for r, g in groups.items():
        n = len(next(iter(g))[1])
        own[r], tight[r] = torch.zeros(n), torch.zeros(n)
        for dep, idxs in g:
            m = dep.target.module
            if not isinstance(m, nn.Conv2d):
                continue
            name, fn = name_of[id(m.weight)], getattr(dep.handler, '__name__', '')
            idx = torch.as_tensor(P._idx_list(idxs), dtype=torch.long)
            W = m.weight.detach().float().cpu()
            if fn == 'prune_out_channels':
                roles['out'][name] = r
                own[r] += own_mass(W, bits)[idx]
            elif fn == 'prune_in_channels':
                roles['in'][name] = r
                if name not in gain_cache:
                    gain_cache[name] = removal_gain(W, bits)
                tight[r] += gain_cache[name][idx]
    return roles, own, tight


def keep_by(scores, cfg):
    """단위마다 점수가 낮은 채널부터 지우고 남는 채널 집합 (trp_joint.materialize와 같은 규칙)."""
    out = {}
    for r, s in scores.items():
        n = len(s)
        out[r] = set(range(n)) - set(torch.argsort(s)[:n - cfg[r]].tolist())
    return out


def z(x):
    return (x - x.mean()) / x.std().clamp_min(1e-12)


def net_metrics(convs, roles, keep, bits):
    """마스크를 적용한 네트워크의 4비트 상대 오차와 crest (필터 파라미터 수 가중 평균)."""
    tp_, te, tc = 0.0, 0.0, 0.0
    for name, W in convs.items():
        o, i = roles['out'].get(name), roles['in'].get(name)
        if o is not None and keep is not None:
            W = W[sorted(keep[o])]
        if i is not None and keep is not None:
            W = W[:, sorted(keep[i])]
        n = (W ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
        rel = qerr(W, bits).sum(1) / n
        crest = W.abs().flatten(1).amax(1) / (n / W[0].numel()).sqrt()
        p = W[0].numel()
        tp_ += p * len(rel)
        te += p * rel.sum().item()
        tc += p * crest.sum().item()
    return te / tp_, tc / tp_


def imp_pct(keep, imp):
    """남긴 채널의 movement 순위 백분위 평균 (단위별 남긴 채널 수로 가중)."""
    tot, cnt = 0.0, 0
    for r, k in keep.items():
        s = imp[r]
        pct = torch.argsort(torch.argsort(s)).float() / max(len(s) - 1, 1)
        tot += pct[sorted(k)].sum().item()
        cnt += len(k)
    return tot / cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.6, 0.8, 0.9])
    ap.add_argument('--bits', type=int, default=4)
    args = ap.parse_args()
    pd.set_option('display.width', 220)

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    cfgs = {r: uniform_cfg(r) for r in args.ratios}
    runs = C.runs_dir(args.work_dir)
    dirs = [d for d in sorted(runs.glob('base/*/fold*_seed0')) if (d / 'result.json').exists() and d.parent.name in tasks]
    rows, ref = [], []
    for n, d in enumerate(dirs, 1):
        task, fold = d.parent.name, int(d.name.split('_')[0][4:])
        base, move = P.load_base(d, len(tasks[task]['classes']), 'cpu'), P.load_movement(d)
        groups = unit_groups(base)
        convs = {k: w.detach().float().cpu() for k, w in C.conv_weights(base).items()}
        roles, own, tight = channel_costs(groups, C.conv_weights(base), args.bits)
        rk_mov, rk_l2 = Ranker.build('mov_cum', base, move), Ranker.build('l2', base, None)
        with torch.no_grad():
            imp = {r: rk_mov.scores(base, g).float() for r, g in groups.items()}
            l2s = {r: rk_l2.scores(base, g).float() for r, g in groups.items()}
        ref.append(net_metrics(convs, roles, None, args.bits)[0])
        for ratio in args.ratios:
            cfg = cfgs[ratio]
            masks = {'movement': keep_by(imp, cfg), 'l2': keep_by(l2s, cfg)}
            for lam in LAMBDAS:
                masks[f'qr_own@{lam}'] = keep_by({r: z(imp[r]) - lam * z(own[r]) for r in imp}, cfg)
                masks[f'qr_full@{lam}'] = keep_by({r: z(imp[r]) - lam * z(own[r] + tight[r]) for r in imp}, cfg)
            for name, keep in masks.items():
                q4, crest = net_metrics(convs, roles, keep, args.bits)
                rows.append({'task': task, 'fold': fold, 'ratio': ratio, 'method': name, 'q4': q4, 'crest': crest,
                             'imp_pct': imp_pct(keep, imp), 'ov_mov': agree(keep, masks['movement']),
                             'ov_l2': agree(keep, masks['l2']), 'silhouette': sim.loc[task, 'silhouette']})
        print(f'[{n}/{len(dirs)}] {task} fold{fold}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(runs / 'qready_signal.csv', index=False)
    mov = df[df.method == 'movement'].set_index(['task', 'fold', 'ratio']).q4
    df['q4_vs_mov'] = df.q4.values / mov.loc[list(zip(df.task, df.fold, df.ratio))].values
    print(f'\n기준 모델 {len(dirs)}개, 결과 표: {runs / "qready_signal.csv"}')
    print(f'자르기 전 원본의 {args.bits}비트 상대 오차 평균: {np.mean(ref):.4f}')
    order = ['movement', 'l2'] + [f'qr_own@{lam}' for lam in LAMBDAS] + [f'qr_full@{lam}' for lam in LAMBDAS]
    verdict = None
    for ratio in args.ratios:
        d = df[df.ratio == ratio]
        tab = d.groupby('method')[['q4', 'q4_vs_mov', 'crest', 'imp_pct', 'ov_mov', 'ov_l2']].mean().loc[order]
        print(f'\n==================== 비율 {ratio} (평균, 기준 모델 {d[["task", "fold"]].drop_duplicates().shape[0]}개) ====================')
        print('q4_vs_mov: movement 마스크 대비 4비트 오차 비 (1보다 작을수록 양자화에 유리)')
        print(tab.round(4).to_string())
        best = d[d.method == 'qr_full@1.0'].copy()
        best['gain'] = 1 - best.q4_vs_mov
        g = best.groupby('task')[['gain', 'silhouette']].mean()
        rho, p = spearmanr(g.silhouette, g.gain)
        print(f'  qr_full@1.0의 오차 감소율과 실루엣의 순위 상관 (과제 평균): {rho:+.2f} (p={p:.3f})')
        if ratio == 0.8:
            verdict = (tab, d)

    if verdict is not None:
        tab, d = verdict
        limit = 0.9 * min(tab.loc['movement', 'q4'], tab.loc['l2', 'q4'])
        ok = [lam for lam in LAMBDAS
              if tab.loc[f'qr_full@{lam}', 'q4'] <= limit and tab.loc[f'qr_full@{lam}', 'imp_pct'] >= tab.loc['l2', 'imp_pct']]
        print(f'\n[판정] 비율 0.8: q4 기준선 {limit:.4f} (movement·l2 중 낮은 쪽의 90%), imp_pct 기준선 '
              f'{tab.loc["l2", "imp_pct"]:.3f} (l2)')
        if ok:
            lam = min(ok, key=lambda x: tab.loc[f'qr_full@{x}', 'q4'])
            print(f'  => 1단계 진행: 조건을 만족하는 λ = {ok}, 추천 λ = {lam} '
                  f'(q4 {tab.loc[f"qr_full@{lam}", "q4"]:.4f}, imp_pct {tab.loc[f"qr_full@{lam}", "imp_pct"]:.3f})')
        else:
            print('  => 1단계 보류: 중요도를 l2만큼 지키면서 양자화 오차를 10% 이상 줄이는 λ가 없다')


if __name__ == '__main__':
    main()
