"""TRP: 궤적 순위·회복 탐침 비균일 프루닝 (Trajectory-ranked, Recovery-Probed pruning).

NetAdapt(ECCV 2018)처럼 매 반복마다 구조 단위별로 후보를 만들어 가장 좋은 하나를 확정하되,
우리 실험 결과에 맞춰 세 가지를 바꿨다.
  1) 단위 안에서 지울 채널: L2 크기 대신 파인튜닝 중 누적 movement 순위 (많이 자를수록 L2보다 유리했다)
  2) 후보 평가: 고정된 배치로 56 step만 회복시킨 뒤 검증 NLL로 고른다.
     자른 직후(BN 재보정만) 점수는 최종 성능을 잘 예측하지 못했다. EagleEye 방식은 trp_bn 변형으로 비교한다.
     고른 후보의 회복된 가중치는 넘기지 않고 구조만 확정한다 -> 최종 모델의 학습량이 비교 대상과 같다.
  3) 단위: DepGraph 그룹 12개(블록 내부 8 + stage별 잔차 채널 4). 자원은 파라미터 수(기본) 또는 MACs.
     단위끼리는 탐침 결과로 비교하므로 층 간 중요도 정규화가 필요 없다.

비교 대상도 여기서 만든다.
  trp_bn        : 탐침 = BN 재보정만 (EagleEye식 평가)
  trp_l2        : 단위 안 순위 = L2 (NetAdapt식 순위)
  global_l2     : Torch-Pruning 전역 프루닝, L2 (예산에 맞게 비율을 이분 탐색)
  global_taylor : 같은 방식, Taylor
균일 프루닝 결과는 prune_grid.py(rec840)에 이미 있다 (예산 0.36/0.16/0.04/0.01 = 비율 0.4/0.6/0.8/0.9).

사용: python trp_search.py --shard 0 --nshards 3
"""
import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torch.utils.data import DataLoader

import common as C
import prune_common as P
from stage_sensitivity import auto_tasks

METHODS = {'trp': ('mov_cum', 'recover'), 'trp_bn': ('mov_cum', 'bn'), 'trp_l2': ('l2', 'recover'),
           'global_l2': ('l2', None), 'global_taylor': ('taylor', None)}
STAGES = ['layer1', 'layer2', 'layer3', 'layer4']
UNIT_ROOTS = [r for s in STAGES for r in (f'{s}.0.conv1', f'{s}.1.conv1', f'{s}.0.conv2')]  # 내부 2 + 잔차 1


def ex_input(model):
    return torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)


def build_dg(model):
    return tp.DependencyGraph().build_dependency(model, example_inputs=ex_input(model))


def resource(model, kind):
    macs, params = tp.utils.count_ops_and_params(model, ex_input(model))
    return float(params if kind == 'params' else macs)


@torch.no_grad()
def conv_spatial(model):
    """conv마다 출력 H*W. 채널을 잘라도 변하지 않으므로 처음에 한 번만 잰다 (MACs 비용 계산용)."""
    sp, hooks = {}, []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=n: sp.__setitem__(n, o.shape[-1] * o.shape[-2])))
    model.eval()(ex_input(model))
    for h in hooks:
        h.remove()
    return sp


def channel_cost(group, name_of, kind, spatial):
    """그룹에서 채널 하나를 지울 때 줄어드는 자원 (파라미터 수 또는 MACs)."""
    cost = 0.0
    for dep, _ in group:
        m = dep.target.module
        fn = getattr(dep.handler, '__name__', '')
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            if fn == 'prune_out_channels':
                per = (m.in_channels // m.groups) * k + (1 if m.bias is not None else 0)
            else:
                per = (m.out_channels // m.groups) * k
            cost += per * (spatial[name_of[m]] if kind == 'macs' else 1)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm) and kind == 'params' and m.affine:
            cost += 2
        elif isinstance(m, nn.Linear) and fn == 'prune_in_channels':
            cost += m.out_features
    return cost


class Ranker:
    """conv 이름별 채널 순위 점수. 채널이 지워지면 점수도 같이 지워서 현재 모델의 인덱스와 맞춘다."""

    def __init__(self, out_scores, in_scores):
        self.out = {n: v.detach().float().cpu().clone() for n, v in out_scores.items()}
        self.inp = {n: v.detach().float().cpu().clone() for n, v in in_scores.items()}

    @classmethod
    def build(cls, rank, base, move):
        if rank == 'mov_cum':
            b = move['best_i']
            return cls({n: v[b] for n, v in move['cum_out'].items()}, {n: v[b] for n, v in move['cum_in'].items()})
        convs = C.conv_weights(base)  # l2: 사전학습 후 파인튜닝된 가중치 크기
        return cls({n: (w ** 2).sum((1, 2, 3)) for n, w in convs.items()},
                   {n: (w ** 2).sum((0, 2, 3)) for n, w in convs.items()})

    def scores(self, model, group):
        return P.ChannelScoreImportance(model, self.out, self.inp)(group)

    def remove(self, group, name_of):
        for dep, idxs in group:
            n = name_of.get(dep.target.module)
            if n not in self.out:
                continue
            fn = getattr(dep.handler, '__name__', '')
            target = self.out if fn == 'prune_out_channels' else self.inp if fn == 'prune_in_channels' else None
            if target is not None:
                keep = torch.ones(len(target[n]), dtype=torch.bool)
                keep[P._idx_list(idxs)] = False
                target[n] = target[n][keep]


def probe_batches(train_ds, n, seed, workers, device):
    """모든 후보가 똑같은 학습 배치로 회복하도록 증강된 배치를 미리 뽑아 GPU에 둔다 (fp16)."""
    g = torch.Generator().manual_seed(seed + 12345)
    loader = DataLoader(train_ds, C.BATCH, shuffle=True, num_workers=workers, generator=g, drop_last=True)
    xs, ys = [], []
    while len(xs) < n:
        for x, y in loader:
            xs.append(x.to(device).half())
            ys.append(y.to(device))
            if len(xs) >= n:
                break
    del loader
    return torch.stack(xs), torch.stack(ys)


def short_train(model, bx, by, steps, seed):
    """탐침용 짧은 회복: 고정 배치, AdamW, 학습률 cosine 1e-4 -> 0."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(bx[i % len(bx)].float()), by[i % len(by)]).backward()
        opt.step()
        sched.step()
    model.eval()


def grads_from_batches(model, bx, by, n):
    model.eval()
    model.zero_grad()
    for i in range(min(n, len(bx))):
        F.cross_entropy(model(bx[i].float()), by[i]).backward()


def evaluate_probe(model, ctx, mode):
    P.recalibrate_bn(model, ctx['cache'], ctx['tr_idx'])
    if mode == 'recover':
        short_train(model, ctx['bx'], ctx['by'], ctx['probe_steps'], ctx['seed'])
    m = C.metrics(C.predict_logits(model, ctx['cache'], ctx['va_idx']), ctx['va_y'])
    return m['nll'], m['acc']


def try_unit(cur, ranker, root, need, kind, spatial, ctx, mode):
    """현재 모델에서 root 단위의 채널을 need만큼 자원이 줄도록 지워 보고 탐침 점수를 잰다."""
    m = copy.deepcopy(cur)
    dg = build_dg(m)
    name_of = {mm: n for n, mm in m.named_modules()}
    mod = dict(m.named_modules())[root]
    full = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=list(range(mod.out_channels)))
    cost = channel_cost(full, name_of, kind, spatial)
    k = math.ceil(need / cost) if cost > 0 else 0
    if k < 1 or mod.out_channels - k < ctx['min_keep']:
        return None
    idxs = torch.argsort(ranker.scores(m, full))[:k].tolist()
    group = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=idxs)
    if not dg.check_pruning_group(group):
        return None
    group.prune()
    nll, acc = evaluate_probe(m, ctx, mode)
    return {'root': root, 'k': k, 'idxs': idxs, 'val_nll': nll, 'val_acc': acc}


def trp_search(base, ranker, kind, budgets, mode, ctx, on_budget):
    """예산(기준 대비 자원 비율)에 닿을 때까지 단위 하나씩 확정한다. 예산마다 on_budget(예산, 현재 모델)을 부른다."""
    cur = copy.deepcopy(base)
    r0 = resource(base, kind)
    spatial = conv_spatial(base)
    trace, n_probes = [], 0
    for b in sorted(budgets, reverse=True):
        while resource(cur, kind) > b * r0 * 1.0001:
            rc = resource(cur, kind)
            need = rc - max(rc * (1 - ctx['step_frac']), b * r0)
            cands = []
            for _ in range(6):  # 어느 단위도 need만큼 줄일 수 없으면 need를 반씩 줄인다
                cands = [c for c in (try_unit(cur, ranker, r, need, kind, spatial, ctx, mode) for r in UNIT_ROOTS) if c]
                if cands:
                    break
                need /= 2
            if not cands:
                return trace, n_probes  # 더 줄일 수 없음: 남은 예산은 건너뛴다
            n_probes += len(cands)
            best = min(cands, key=lambda c: (c['val_nll'], -c['val_acc']))
            dg = build_dg(cur)
            name_of = {mm: n for n, mm in cur.named_modules()}
            group = dg.get_pruning_group(dict(cur.named_modules())[best['root']], tp.prune_conv_out_channels,
                                         idxs=best['idxs'])
            ranker.remove(group, name_of)
            group.prune()
            trace.append({'unit': best['root'], 'k': best['k'], 'val_nll': best['val_nll'],
                          'val_acc': best['val_acc'], 'resource_ratio': resource(cur, kind) / r0,
                          'n_cands': len(cands)})
        on_budget(b, cur, trace, n_probes)
    return trace, n_probes


def global_search(base, crit, kind, budget, ctx):
    """전역 프루닝 비율을 이분 탐색해서 예산 이하에서 가장 큰 모델을 찾는다."""
    r0 = resource(base, kind)
    lo, hi, best = 0.0, 0.99, None
    for _ in range(14):
        mid = (lo + hi) / 2
        m = copy.deepcopy(base)
        if crit == 'taylor':
            grads_from_batches(m, ctx['bx'], ctx['by'], 20)
        P.prune_global(m, P.make_importance(crit, m, None), mid)
        r = resource(m, kind) / r0
        if r > budget:
            lo = mid
        else:
            hi, best = mid, (m, mid)
    return best


def finalize(model, ctx, out, info):
    """예산 모델을 다른 실험과 똑같이 BN 재보정 + 840 step 회복한 뒤 평가해 저장한다."""
    C.set_seed(ctx['seed'])
    P.recalibrate_bn(model, ctx['cache'], ctx['tr_idx'])
    P.recover(model, ctx['train_ds'], ctx['final_steps'], ctx['device'], ctx['workers'], ctx['seed'])
    val_logits = C.predict_logits(model, ctx['cache'], ctx['va_idx'])
    test_logits = C.predict_logits(model, ctx['cache'], ctx['te_idx'])
    st = P.model_stats(model)
    info.update({'val': C.metrics(val_logits, ctx['va_y']), 'test': C.metrics(test_logits, ctx['te_y']),
                 'params': st['params'], 'macs': st['macs'],
                 'params_ratio': st['params'] / ctx['base_stats']['params'],
                 'macs_ratio': st['macs'] / ctx['base_stats']['macs'], 'channels': st['channels']})
    P.save_result(out, info, {'test_idx': ctx['te_idx'], 'test_y': ctx['te_y'], 'test_logits': test_logits.numpy()})
    print(f"  {info['method']} b={info['budget']}: test acc={info['test']['acc']:.4f} "
          f"params={info['params_ratio']:.3f} macs={info['macs_ratio']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--tasks', default='auto', help="'auto'(6클래스 + 가장 어려운/쉬운 쌍·삼중) 또는 쉼표로 구분")
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    ap.add_argument('--budgets', type=float, nargs='+', default=[0.36, 0.16, 0.04, 0.01],
                    help='기준 모델 대비 남길 자원 비율 (균일 0.4/0.6/0.8/0.9와 같은 크기)')
    ap.add_argument('--resource', choices=['params', 'macs'], default='params')
    ap.add_argument('--probe-steps', type=int, default=56)
    ap.add_argument('--final-steps', type=int, default=840)
    ap.add_argument('--step-frac', type=float, default=0.2, help='반복마다 현재 자원의 몇 %%를 줄일지')
    ap.add_argument('--min-keep', type=int, default=4, help='그룹마다 최소로 남길 채널 수')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='trp')
    args = ap.parse_args()

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    names = auto_tasks(args.work_dir) if args.tasks == 'auto' else args.tasks.split(',')
    jobs = [(t, f, s) for t in names for f in args.folds for s in args.seeds]
    mine = jobs[args.shard::args.nshards]
    print(f'과제 {names}\n기준 모델 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개, 방법 {args.methods}')

    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for n, (t, f, s) in enumerate(mine, 1):
        out_dir = runs / args.out_name / t / f'fold{f}_seed{s}'
        todo = [m for m in args.methods if not all((out_dir / f'{m}_b{b}.json').exists() for b in args.budgets)]
        if not todo:
            continue
        t0 = time.time()
        task = tasks[t]
        split = C.task_split(images, task, f)
        (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
        base_dir = runs / 'base' / t / f'fold{f}_seed{s}'
        base = P.load_base(base_dir, len(task['classes']), args.device)
        move = P.load_movement(base_dir)
        train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
        bx, by = probe_batches(train_ds, args.probe_steps, s, args.workers, args.device)
        ctx = {'cache': cache, 'tr_idx': tr_idx, 'va_idx': va_idx, 'va_y': va_y, 'te_idx': te_idx, 'te_y': te_y,
               'train_ds': train_ds, 'bx': bx, 'by': by, 'seed': s, 'device': args.device, 'workers': args.workers,
               'probe_steps': args.probe_steps, 'final_steps': args.final_steps, 'step_frac': args.step_frac,
               'min_keep': args.min_keep, 'base_stats': P.model_stats(base)}
        common_info = {'task': t, 'fold': f, 'seed': s, 'resource': args.resource,
                       'probe_steps': args.probe_steps, 'final_steps': args.final_steps}
        print(f'[{n}/{len(mine)}] {t} fold{f} seed{s}', flush=True)

        for method in todo:
            rank, mode = METHODS[method]
            tm = time.time()
            if method.startswith('global'):
                for b in args.budgets:
                    out = out_dir / f'{method}_b{b}.json'
                    if out.exists():
                        continue
                    found = global_search(base, rank, args.resource, b, ctx)
                    if found is None:
                        print(f'  {method} b={b}: 예산까지 줄일 수 없음')
                        continue
                    model, ratio = found
                    finalize(model, ctx, out, {**common_info, 'method': method, 'budget': b, 'global_ratio': ratio,
                                               'search_sec': round(time.time() - tm, 1)})
                continue

            def on_budget(b, cur, trace, n_probes, method=method, rank=rank, mode=mode, tm=tm):
                out = out_dir / f'{method}_b{b}.json'
                if out.exists():
                    return
                finalize(copy.deepcopy(cur), ctx, out,
                         {**common_info, 'method': method, 'budget': b, 'rank': rank, 'probe': mode,
                          'n_probes': n_probes, 'probe_train_steps': n_probes * (args.probe_steps if mode == 'recover' else 0),
                          'iterations': len(trace), 'search_sec': round(time.time() - tm, 1)})

            trace, n_probes = trp_search(base, Ranker.build(rank, base, move), args.resource, args.budgets,
                                         mode, ctx, on_budget)
            P.save_result(out_dir / f'{method}_trace.json', {'trace': trace, 'n_probes': n_probes})
        print(f'  완료 ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
