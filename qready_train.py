"""1단계 (양자화 대비 프루닝): 0단계의 마스크로 실제로 자르고 회복한 뒤, 정확도와 단순 양자화 후 정확도를 비교한다.

같은 너비(균일 비율)에서 남길 채널만 다르다. 마스크는 qready_signal.py와 똑같이 기준 모델에서 계산한다.
  비율 0.8: movement, l2, qr_full@1.0, qr_full@4.0
  비율 0.9: movement, qr_full@1.0
회복은 지금까지와 같다 (BN 재보정 -> AdamW 1e-4 cosine 2800 step). 자른 모델은 팀원 양자화(PPO) 비교용으로 저장한다.

양자화는 팀 방식이 정해지기 전의 측정 도구로만 쓴다: conv 가중치만, 출력 채널별 대칭, 최댓값 눈금, 반올림, BN 그대로.
  W8   : 모든 conv 필터 8비트
  W4   : 모든 conv 필터 4비트
  mix5 : 모두 4비트에서 시작해 4비트 오차가 큰 필터부터 8비트로 올려 conv 가중치 평균 5비트를 맞춘다
         (같은 너비면 크기가 같으므로 '같은 크기 예산'에서의 비교가 된다)

판정 규칙 (결과를 보기 전에 정함, 비율 0.8):
  어떤 λ에서 (a) 원본 정밀도 test 정확도 차이(qr - movement)가 평균 -0.5%p 이상이고
  (b) mix5 test 정확도 차이가 양수(Wilcoxon p<0.05)이면서 원본 정밀도 차이보다 크면 '효과 확인'

사용: python qready_train.py --smoke
      python qready_train.py --shard 0 --nshards 3
      python qready_train.py --summary > work/qready_summary.txt
"""
import argparse
import copy
import json
import time

import pandas as pd
import torch
import torch.nn as nn
import torch_pruning as tp
from scipy.stats import spearmanr

import common as C
import prune_common as P
from early_signal import uniform_cfg, unit_groups
from qready_signal import channel_costs, keep_by, net_metrics, qerr, z
from summarize_budget import fmt, stat
from trp_search import UNIT_ROOTS, Ranker, build_dg

ARMS = {0.8: ['movement', 'l2', 'qr_full@1.0', 'qr_full@4.0'], 0.9: ['movement', 'qr_full@1.0']}
QCONF = ['fp32', 'W8', 'W4', 'mix5']


def masks_for(base, move, cfg, arms, bits=4):
    """qready_signal.py와 같은 규칙으로 방법별 '단위마다 남길 채널' 집합을 만든다."""
    groups = unit_groups(base)
    _, own, tight = channel_costs(groups, C.conv_weights(base), bits)
    rk_mov, rk_l2 = Ranker.build('mov_cum', base, move), Ranker.build('l2', base, None)
    with torch.no_grad():
        imp = {r: rk_mov.scores(base, g).float() for r, g in groups.items()}
        l2s = {r: rk_l2.scores(base, g).float() for r, g in groups.items()}
    out = {}
    for arm in arms:
        if arm == 'movement':
            s = imp
        elif arm == 'l2':
            s = l2s
        else:
            lam = float(arm.split('@')[1])
            s = {r: z(imp[r]) - lam * z(own[r] + tight[r]) for r in imp}
        out[arm] = keep_by(s, cfg)
    return out


def materialize_keep(base, keep):
    """단위별 남길 채널대로 실제로 자른다. 단위끼리 채널 차원이 겹치지 않아 원래 번호를 그대로 쓴다."""
    m = copy.deepcopy(base)
    dg, mods = build_dg(m), dict(m.named_modules())
    for r in UNIT_ROOTS:
        drop = sorted(set(range(mods[r].out_channels)) - keep[r])
        if drop:
            dg.get_pruning_group(mods[r], tp.prune_conv_out_channels, idxs=drop).prune()
    return m


def bit_plans(model, avg_bits=5.0):
    """필터별 비트: W8, W4, mix5 (4비트 상대 오차가 큰 필터부터 8비트로 올려 평균 avg_bits)."""
    convs = {n: m.weight.detach().float().cpu() for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    plans = {b: {n: torch.full((W.shape[0],), int(b[1:])) for n, W in convs.items()} for b in ('W8', 'W4')}
    items = []
    for n, W in convs.items():
        rel = qerr(W, 4).sum(1) / (W ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
        items += [(r, n, j, W[0].numel()) for j, r in enumerate(rel.tolist())]
    budget = (avg_bits - 4) * sum(p for *_, p in items)
    mix = {n: torch.full((W.shape[0],), 4) for n, W in convs.items()}
    for _, n, j, p in sorted(items, reverse=True):
        if 4 * p <= budget:
            mix[n][j] = 8
            budget -= 4 * p
    plans['mix5'] = mix
    return plans


@torch.no_grad()
def quantize_convs(model, plan):
    """conv 가중치를 필터별 비트로 대칭 양자화한 사본 (fc와 BN은 그대로)."""
    q = copy.deepcopy(model)
    for name, m in q.named_modules():
        if isinstance(m, nn.Conv2d):
            W = m.weight.data
            qmax = (2 ** (plan[name].to(W.device).float() - 1) - 1).view(-1, 1, 1, 1)
            s = (W.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax).clamp_min(1e-12)
            m.weight.data = torch.minimum(torch.maximum((W / s).round(), -qmax), qmax) * s
    return q


def evaluate(model, cache, split):
    return {k: C.metrics(C.predict_logits(model, cache, split[k][0]), split[k][1]) for k in ('val', 'test')}


def run_condition(args, images, tasks, cache, task, fold, out_root, arms):
    base_dir = C.runs_dir(args.work_dir) / 'base' / task / f'fold{fold}_seed0'
    todo = [(r, a) for r in args.ratios for a in arms[r]
            if not (out_root / task / f'fold{fold}_seed0' / f'{a}_r{r}.json').exists()]
    if not todo:
        return
    base, move = P.load_base(base_dir, len(tasks[task]['classes']), args.device), P.load_movement(base_dir)
    split = C.task_split(images, tasks[task], fold)
    tr_idx, tr_y = split['train']
    train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    masks = {r: masks_for(base, move, uniform_cfg(r), arms[r]) for r in sorted({r for r, _ in todo})}
    for r, arm in todo:
        t0 = time.time()
        model = materialize_keep(base, masks[r][arm])
        C.set_seed(0)
        P.recalibrate_bn(model, cache, tr_idx)
        P.recover(model, train_ds, args.steps, args.device, args.workers, 0)
        res = {'fp32': evaluate(model, cache, split)}
        for qname, plan in bit_plans(model).items():
            res[qname] = evaluate(quantize_convs(model, plan), cache, split)
        convs = {n: m.weight.detach().float().cpu() for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
        q4, crest = net_metrics(convs, {'out': {}, 'in': {}}, None, 4)
        st = P.model_stats(model)
        info = {'task': task, 'fold': fold, 'ratio': r, 'arm': arm, 'steps': args.steps, 'results': res,
                'q4_after': q4, 'crest_after': crest, 'params': st['params'], 'macs': st['macs'],
                'time_sec': round(time.time() - t0, 1)}
        out = out_root / task / f'fold{fold}_seed0' / f'{arm}_r{r}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.cpu(), out.with_suffix('.pt'))  # 팀원 양자화 비교용 (구조째 저장)
        out.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
        print(f"{task} f{fold} r{r} {arm}: fp32 {res['fp32']['test']['acc']:.4f} W4 {res['W4']['test']['acc']:.4f} "
              f"mix5 {res['mix5']['test']['acc']:.4f} q4 {q4:.4f} ({info['time_sec']}s)", flush=True)
        del model


def summary(args):
    rows = []
    root = C.runs_dir(args.work_dir) / args.out_name
    for p in root.glob('*/fold*_seed0/*.json'):
        j = json.loads(p.read_text(encoding='utf-8'))
        row = {'task': j['task'], 'fold': j['fold'], 'ratio': j['ratio'], 'arm': j['arm'],
               'q4_after': j['q4_after'], 'crest_after': j['crest_after']}
        row.update({q: 100 * j['results'][q]['test']['acc'] for q in QCONF})
        rows.append(row)
    if not rows:
        raise SystemExit('결과가 없습니다.')
    df = pd.DataFrame(rows)
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    pd.set_option('display.width', 220)
    print(f'=== 양자화 대비 프루닝 1단계 (test 정확도 %, 회복 {args.steps} step) ===')
    print('양자화는 측정용 대리 방식: conv만, 출력 채널별 대칭, W8/W4는 전부 8/4비트, mix5는 같은 크기(평균 5비트)')
    verdict = []
    for r in sorted(df.ratio.unique()):
        d = df[df.ratio == r]
        print(f'\n==================== 비율 {r} ====================')
        tab = d.groupby('arm')[QCONF + ['q4_after', 'crest_after']].mean()
        tab['n'] = d.groupby('arm').size()
        print(tab.loc[[a for a in ARMS.get(r, tab.index) if a in tab.index]].round(4).to_string())
        w = d.pivot_table(index=['task', 'fold'], columns='arm', values=QCONF)
        print('\n  movement 대비 차이 (%p, 같은 과제·fold): 평균 [승/무/패] Wilcoxon p')
        for arm in [a for a in ARMS.get(r, []) if a != 'movement' and (QCONF[0], a) in w.columns]:
            diffs = {q: (w[(q, arm)] - w[(q, 'movement')]).dropna() for q in QCONF}
            gain = (diffs['mix5'] - diffs['fp32']).dropna()
            print(f'  {arm}')
            for q in QCONF:
                print(f'    {q:<5} {fmt(*stat(diffs[q]))}  n={len(diffs[q])}')
            print(f'    양자화 이득 (mix5 차이 - fp32 차이) {fmt(*stat(gain))}')
            g = diffs['fp32'].groupby(level='task').mean()
            rho, p = spearmanr(sim.loc[g.index, 'silhouette'], g)
            print(f'    원본 정밀도 차이와 실루엣의 순위 상관 (과제 평균): {rho:+.2f} (p={p:.3f})')
            if r == 0.8:
                m_fp, _, _ = stat(diffs['fp32'])
                m_mix, _, p_mix = stat(diffs['mix5'])
                verdict.append((arm, m_fp >= -0.5 and m_mix > 0 and p_mix < 0.05 and m_mix > m_fp, m_fp, m_mix, p_mix))
    if verdict:
        print('\n[판정] 비율 0.8 (규칙은 파일 맨 위 설명)')
        for arm, ok, m_fp, m_mix, p_mix in verdict:
            print(f'  {arm}: 원본 정밀도 {m_fp:+.2f}%p, mix5 {m_mix:+.2f}%p (p={p_mix:.3f}) -> {"충족" if ok else "미충족"}')
        print('  => ' + ('효과 확인: 같은 너비의 movement 모델과 함께 팀원 PPO 비교에 넘긴다' if any(v[1] for v in verdict)
                         else '효과 없음: 가중치 수준의 양자화 친화성이 정확도로 이어지지 않았다'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.8, 0.9])
    ap.add_argument('--steps', type=int, default=2800)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='qready')
    ap.add_argument('--arms', nargs='+', default=None,
                    help="'비율:방법' 목록 (예: 0.4:movement 0.6:movement). 주면 기본 비교군과 --ratios 대신 쓴다")
    ap.add_argument('--smoke', action='store_true', help='pair_gl-pl fold0, 16 step (기본: 비율 0.8 movement·qr_full@1.0)')
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()
    if args.summary:
        summary(args)
        return
    arms = ARMS
    if args.arms:
        arms = {}
        for item in args.arms:
            r, a = item.split(':')
            arms.setdefault(float(r), []).append(a)
        args.ratios = sorted(arms)
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    cache = C.EvalCache(args.work_dir, args.device)
    if args.smoke:
        if not args.arms:
            arms = {0.8: ['movement', 'qr_full@1.0']}
        args.ratios, args.steps = sorted(arms), 16
        run_condition(args, images, tasks, cache, 'pair_gl-pl', 0, C.runs_dir(args.work_dir) / (args.out_name + '_smoke'),
                      arms)
        return
    names = list(tasks) if args.tasks == 'all' else args.tasks.split(',')
    conds = [(t, f) for t in names for f in args.folds]
    mine = conds[args.shard::args.nshards]
    print(f'조건 {len(conds)}개 중 이 프로세스 {len(mine)}개, 조건마다 {sum(len(arms[r]) for r in args.ratios)}개 실행')
    for n, (t, f) in enumerate(mine, 1):
        run_condition(args, images, tasks, cache, t, f, C.runs_dir(args.work_dir) / args.out_name, arms)
        print(f'[{n}/{len(mine)}] {t} fold{f} 완료', flush=True)


if __name__ == '__main__':
    main()
