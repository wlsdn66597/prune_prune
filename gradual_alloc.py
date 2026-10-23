"""배분 + 점진 결합 실험: TRP(또는 TRP-J) 배분을 목표로 GTP처럼 나눠 자르고, movement를 회복 중에도 갱신한다.

지금까지의 결과에서 남은 두 질문을 확인한다.
  1) GTP가 one-shot에 진 이유가 '최종 구조로 학습하는 시간이 짧아서'인가?
     -> 목표 도달 시점을 전체 840 step의 60%에서 30%로 앞당긴다 (e30 vs e60)
  2) GTP에서 큰 효과를 보인 'movement 갱신'이 TRP 배분 위에서도 더해지는가?
     -> TRP 배분을 목표로 나눠 자르면서 갱신(online)과 고정(fixed)을 비교한다
목표 배분: trp (trp_bn의 파라미터 예산 결과), joint (TRP-J: 균일과 파라미터·MACs가 같거나 작음), uniform (균일).
비율 키는 균일 비율로 통일한다 (TRP 예산 0.36/0.16/0.04/0.01 = 균일 0.4/0.6/0.8/0.9와 같은 크기).
전체 학습량은 one-shot 회복과 같은 840 step이고, 목표까지 6번에 나눠 cubic 일정으로 자른다.

사용: python gradual_alloc.py --target trp --end-frac 0.3 --shard 0 --nshards 3
"""
import argparse
import copy
import json
import math
import time

import torch
import torch.nn.functional as F
import torch_pruning as tp
from torch.utils.data import DataLoader

import common as C
import prune_common as P
from gradual_prune import cycle, zeros_like_sums
from trp_search import UNIT_ROOTS, Ranker, build_dg

BUDGET = {0.4: 0.36, 0.6: 0.16, 0.8: 0.04, 0.9: 0.01}  # 균일 비율 -> 같은 크기의 TRP 파라미터 예산


def config_of(model):
    mods = dict(model.named_modules())
    return {r: mods[r].out_channels for r in UNIT_ROOTS}


def load_target(runs, target, t, f, s, r, orig, trp_method):
    """목표 배분 {단위 이름: 남길 채널 수}. 해당 결과 파일이 없으면 None."""
    if target == 'uniform':
        return {k: max(1, int(orig[k] * (1 - r))) for k in UNIT_ROOTS}
    d = runs / ('trp' if target == 'trp' else 'joint') / t / f'fold{f}_seed{s}'
    p = d / (f'{trp_method}_b{BUDGET[r]}.json' if target == 'trp' else f'trp_joint_r{r}.json')
    if not p.exists():
        return None
    info = json.loads(p.read_text(encoding='utf-8'))
    if target == 'trp':
        return {k: int(info['channels'][k]) for k in UNIT_ROOTS}
    return {k: int(v) for k, v in info['cfg'].items()}


def prune_toward(model, ranker, frac, orig, target):
    """단위마다 원래 너비에서 목표까지의 frac만큼 줄인다 (frac=1이면 목표와 같음)."""
    dg = build_dg(model)
    name_of = {m: n for n, m in model.named_modules()}
    mods = dict(model.named_modules())
    for root in UNIT_ROOTS:
        mod = mods[root]
        keep = target[root] + int((orig[root] - target[root]) * (1 - frac))
        k = mod.out_channels - keep
        if k <= 0:
            continue
        full = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=list(range(mod.out_channels)))
        idxs = torch.argsort(ranker.scores(model, full))[:k].tolist()
        group = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=idxs)
        if dg.check_pruning_group(group):
            ranker.remove(group, name_of)
            group.prune()


def run(base, move, target, update, ctx):
    model = copy.deepcopy(base)
    orig = config_of(model)
    ranker = Ranker.build('mov_cum', base, move)
    T, end, dev = ctx['steps'], ctx['prune_end'], ctx['device']
    every = max(1, end // ctx['n_events'])
    events = set(range(every, end + 1, every)) | {end}
    g = torch.Generator().manual_seed(ctx['seed'])
    loader = DataLoader(ctx['train_ds'], C.BATCH, shuffle=True, num_workers=ctx['workers'], generator=g,
                        pin_memory=dev != 'cpu', persistent_workers=ctx['workers'] > 0)
    batches = cycle(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR)
    acc = zeros_like_sums(model) if update else None
    for step in range(T):
        if step in events:
            if update:  # 이번 구간에 쌓인 movement를 순위 점수에 더한다
                for n in acc[0]:
                    ranker.out[n] += acc[0][n].cpu()
                    ranker.inp[n] += acc[1][n].cpu()
            prune_toward(model, ranker, 1 - (1 - step / end) ** 3, orig, target)
            opt = torch.optim.AdamW(model.parameters(), lr=C.LR)
            if update:
                acc = zeros_like_sums(model)
        for grp in opt.param_groups:  # one-shot 회복과 같은 cosine 일정 (1e-4 -> 0)
            grp['lr'] = 0.5 * C.LR * (1 + math.cos(math.pi * step / T))
        x, y = next(batches)
        model.train()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x.to(dev, non_blocking=True)), y.to(dev, non_blocking=True)).backward()
        if update:
            with torch.no_grad():
                for n, w in C.conv_weights(model).items():
                    gw = -(w.grad * w)
                    acc[0][n] += gw.sum((1, 2, 3))
                    acc[1][n] += gw.sum((0, 2, 3))
        opt.step()
    del loader
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--target', choices=['trp', 'joint', 'uniform'], default='trp')
    ap.add_argument('--trp-method', default='trp_bn', help='목표 배분으로 쓸 TRP 결과 (trp_bn 또는 trp)')
    ap.add_argument('--fixed', action='store_true', help='movement를 갱신하지 않고 기준 모델의 값으로 고정')
    ap.add_argument('--end-frac', type=float, default=0.3, help='전체 step 중 이 지점에서 목표 배분에 도달')
    ap.add_argument('--n-events', type=int, default=6, help='목표까지 나눠 자르는 횟수')
    ap.add_argument('--tasks', default='auto')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.4, 0.6, 0.8, 0.9])
    ap.add_argument('--steps', type=int, default=840)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='gradalloc')
    args = ap.parse_args()

    from stage_sensitivity import auto_tasks
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    names = auto_tasks(args.work_dir) if args.tasks == 'auto' else args.tasks.split(',')
    method = f"grad_{args.target}_{'fixed' if args.fixed else 'online'}_e{round(args.end_frac * 100)}"
    jobs = [(t, f, s) for t in names for f in args.folds for s in args.seeds]
    mine = jobs[args.shard::args.nshards]
    print(f'방법 {method}, 과제 {names}\n기준 모델 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개')

    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for n, (t, f, s) in enumerate(mine, 1):
        out_dir = runs / args.out_name / t / f'fold{f}_seed{s}'
        todo = [r for r in args.ratios if not (out_dir / f'{method}_r{r}.json').exists()]
        if not todo:
            continue
        t0 = time.time()
        task = tasks[t]
        split = C.task_split(images, task, f)
        (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
        base_dir = runs / 'base' / t / f'fold{f}_seed{s}'
        base = P.load_base(base_dir, len(task['classes']), args.device)
        move = P.load_movement(base_dir)
        base_stats = P.model_stats(base)
        orig = config_of(base)
        ctx = {'train_ds': C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf), 'seed': s,
               'device': args.device, 'workers': args.workers, 'steps': args.steps,
               'prune_end': int(args.steps * args.end_frac), 'n_events': args.n_events}
        for r in todo:
            target = load_target(runs, args.target, t, f, s, r, orig, args.trp_method)
            if target is None:
                print(f'  {t} f{f} r={r}: 목표 배분 결과가 없어 건너뜀')
                continue
            C.set_seed(s)
            model = run(base, move, target, not args.fixed, ctx)
            val_logits = C.predict_logits(model, cache, va_idx)
            test_logits = C.predict_logits(model, cache, te_idx)
            st = P.model_stats(model)
            info = {'task': t, 'fold': f, 'seed': s, 'method': method, 'ratio': r, 'target': args.target,
                    'update': not args.fixed, 'end_frac': args.end_frac, 'n_events': args.n_events,
                    'steps': args.steps, 'target_cfg': target,
                    'val': C.metrics(val_logits, va_y), 'test': C.metrics(test_logits, te_y),
                    'params': st['params'], 'macs': st['macs'],
                    'params_ratio': st['params'] / base_stats['params'],
                    'macs_ratio': st['macs'] / base_stats['macs'], 'channels': st['channels']}
            P.save_result(out_dir / f'{method}_r{r}.json', info,
                          {'test_idx': te_idx, 'test_y': te_y, 'test_logits': test_logits.numpy()})
            print(f"  {t} f{f} r={r}: test acc={info['test']['acc']:.4f} params={info['params_ratio']:.3f} "
                  f"macs={info['macs_ratio']:.3f}", flush=True)
        print(f'[{n}/{len(mine)}] {t} fold{f} seed{s} 완료 ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
