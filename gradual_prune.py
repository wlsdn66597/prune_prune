"""GTP: 점진 궤적 프루닝 (Gradual Trajectory Pruning).

TRP가 '어디를 자를지'를 탐색했다면, GTP는 '언제 자를지'를 바꾼다.
같은 회복 예산(840 step) 안에서 채널을 한 번에 자르지 않고 cubic 일정(Zhu & Gupta 2017)에 따라
조금씩 자르면서, movement를 회복 학습 중에도 계속 누적해 순위를 갱신한다.
  - 자른 직후 상태만 보고 한 번에 정하면 틀리기 쉽다 (발견 1) -> 학습하면서 나눠서 정한다
  - movement는 파인튜닝 궤적 정보다 (발견 2) -> 자른 뒤 회복하는 궤적도 계속 반영한다
  - 체크포인트로 사후 근사할 수 없다 (발견 5) -> 학습 중에 바로 누적한다

변형
  gtp        : 점진 + movement 계속 누적 (기준 모델의 movement에서 시작)
  gtp_fixed  : 점진 + 기준 모델의 movement 고정 (누적 갱신의 효과를 분리)
  gtp_l2     : 점진 + 자르는 시점의 L2 크기 (채널 단위 GMP)
비교 대상인 같은 비율의 one-shot(oneshot_mov = mov_cum, oneshot_l2 = l2)은 prune_grid.py rec840에 이미 있다.
모든 그룹을 같은 비율로 자르므로 최종 채널 수가 one-shot과 같다.

사용: python gradual_prune.py --shard 0 --nshards 3
"""
import argparse
import copy
import math
import time

import torch
import torch.nn.functional as F
import torch_pruning as tp
from torch.utils.data import DataLoader

import common as C
import prune_common as P
from trp_search import UNIT_ROOTS, Ranker, build_dg

METHODS = ['gtp', 'gtp_fixed', 'gtp_l2']


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def prune_to(model, ranker, s, orig, min_keep):
    """모든 그룹을 원래 너비의 (1 - s)만 남도록 자른다 (Torch-Pruning 균일 프루닝처럼 내림)."""
    dg = build_dg(model)
    name_of = {m: n for n, m in model.named_modules()}
    mods = dict(model.named_modules())
    for root in UNIT_ROOTS:
        mod = mods[root]
        k = mod.out_channels - max(min_keep, int(orig[root] * (1 - s)))
        if k <= 0:
            continue
        full = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=list(range(mod.out_channels)))
        idxs = torch.argsort(ranker.scores(model, full))[:k].tolist()
        group = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=idxs)
        if dg.check_pruning_group(group):
            ranker.remove(group, name_of)
            group.prune()


def zeros_like_sums(model):
    convs = C.conv_weights(model)
    return ({n: torch.zeros(w.shape[0], device=w.device) for n, w in convs.items()},
            {n: torch.zeros(w.shape[1], device=w.device) for n, w in convs.items()})


def run_gradual(base, move, method, ratio, ctx):
    model = copy.deepcopy(base)
    orig = {r: dict(model.named_modules())[r].out_channels for r in UNIT_ROOTS}
    ranker = Ranker.build('mov_cum', base, move) if method in ('gtp', 'gtp_fixed') else None
    update = method == 'gtp'
    T, every, end, dev = ctx['steps'], ctx['prune_every'], ctx['prune_end'], ctx['device']
    events = set(range(every, end + 1, every))
    g = torch.Generator().manual_seed(ctx['seed'])
    loader = DataLoader(ctx['train_ds'], C.BATCH, shuffle=True, num_workers=ctx['workers'], generator=g,
                        pin_memory=dev != 'cpu', persistent_workers=ctx['workers'] > 0)
    batches = cycle(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR)
    acc = zeros_like_sums(model) if update else None
    for step in range(T):
        if step in events:
            s = ratio * (1 - (1 - step / end) ** 3)
            if update:  # 이번 구간에 쌓인 movement를 순위 점수에 더한다
                for n in acc[0]:
                    ranker.out[n] += acc[0][n].cpu()
                    ranker.inp[n] += acc[1][n].cpu()
            prune_to(model, ranker if ranker is not None else Ranker.build('l2', model, None), s, orig,
                     ctx['min_keep'])
            opt = torch.optim.AdamW(model.parameters(), lr=C.LR)  # 모양이 바뀐 파라미터로 다시 만든다
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
    ap.add_argument('--tasks', default='all', help="'all', 'auto'(5과제) 또는 쉼표로 구분한 과제 이름")
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.6, 0.8, 0.9])
    ap.add_argument('--steps', type=int, default=840, help='전체 학습량 (one-shot 회복과 같게)')
    ap.add_argument('--prune-every', type=int, default=56)
    ap.add_argument('--prune-end-frac', type=float, default=0.6, help='전체 step 중 이 지점까지 목표 비율에 도달')
    ap.add_argument('--min-keep', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='gradual')
    args = ap.parse_args()

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    if args.tasks == 'all':
        names = list(tasks)
    elif args.tasks == 'auto':
        from stage_sensitivity import auto_tasks
        names = auto_tasks(args.work_dir)
    else:
        names = args.tasks.split(',')
    jobs = [(t, f, s) for t in names for f in args.folds for s in args.seeds]
    mine = jobs[args.shard::args.nshards]
    print(f'기준 모델 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개, 방법 {args.methods} x 비율 {args.ratios}')

    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for n, (t, f, s) in enumerate(mine, 1):
        out_dir = runs / args.out_name / t / f'fold{f}_seed{s}'
        todo = [(m, r) for m in args.methods for r in args.ratios if not (out_dir / f'{m}_r{r}.json').exists()]
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
        ctx = {'train_ds': C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf), 'seed': s,
               'device': args.device, 'workers': args.workers, 'steps': args.steps,
               'prune_every': args.prune_every, 'prune_end': int(args.steps * args.prune_end_frac),
               'min_keep': args.min_keep}
        for method, r in todo:
            C.set_seed(s)
            model = run_gradual(base, move, method, r, ctx)
            val_logits = C.predict_logits(model, cache, va_idx)
            test_logits = C.predict_logits(model, cache, te_idx)
            st = P.model_stats(model)
            info = {'task': t, 'fold': f, 'seed': s, 'method': method, 'ratio': r, 'steps': args.steps,
                    'prune_every': args.prune_every, 'prune_end': ctx['prune_end'],
                    'val': C.metrics(val_logits, va_y), 'test': C.metrics(test_logits, te_y),
                    'params': st['params'], 'macs': st['macs'],
                    'params_ratio': st['params'] / base_stats['params'],
                    'macs_ratio': st['macs'] / base_stats['macs'], 'channels': st['channels']}
            P.save_result(out_dir / f'{method}_r{r}.json', info,
                          {'test_idx': te_idx, 'test_y': te_y, 'test_logits': test_logits.numpy()})
            print(f"  {t} f{f} {method} r={r}: test acc={info['test']['acc']:.4f} "
                  f"params={info['params_ratio']:.3f}", flush=True)
        print(f'[{n}/{len(mine)}] {t} fold{f} seed{s} 완료 ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
