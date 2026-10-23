"""TRP-J: 크기(파라미터)와 속도(MACs)를 함께 제약하는 TRP.

MACs 예산 실험에서 확인한 두 가지를 반영했다.
  - ResNet-18은 MACs가 stage마다 고르게 퍼져 있어서, 연산량 기준에서는 균일 배분이 이미 거의 최적이었다.
  - 탐욕 탐색은 매 단계 손실이 작아 보이는 잔차 채널을 계속 골라, 많이 줄일수록 잔차 채널이 지나치게 얇아졌다.
그래서
  1) 파라미터와 MACs를 동시에 제약한다. 예산 = 같은 비율로 자른 균일 모델의 파라미터와 MACs.
     즉 결과 모델은 균일 모델보다 크기도 연산량도 크지 않다.
  2) 시작점 후보는 균일 배분과, 두 제약을 함께 거는 탐욕 TRP. 56 step 회복 탐침으로 더 나은 쪽을 고른다.
  3) 시작점에서 '한 단위를 늘리고 다른 단위를 줄이는' 재배분을 BN 재보정 탐침으로 평가해, 나아질 때만 받아들인다.
     (탐욕 방식은 한 번 자른 채널을 되살리지 못했지만, 재배분은 얇아진 단위를 다시 늘릴 수 있다)
  4) 마지막에 시작점과 재배분 결과를 56 step 회복 탐침으로 다시 비교해 고른다 (검증셋 과적합 방지).

결과 파일: joint_greedy_r{비율}.json (2번 탐욕 결과), trp_joint_r{비율}.json (최종).
같은 비율의 균일 결과(uniform_mov)는 prune_grid.py rec840에 있다.

사용: python trp_joint.py --shard 0 --nshards 3
"""
import argparse
import copy
import math
import random
import time

import torch
import torch_pruning as tp

import common as C
import prune_common as P
from stage_sensitivity import auto_tasks
from trp_search import (UNIT_ROOTS, Ranker, build_dg, channel_cost, conv_spatial, evaluate_probe, ex_input,
                        finalize, probe_batches, try_unit)


def resources(model):
    macs, params = tp.utils.count_ops_and_params(model, ex_input(model))
    return float(params), float(macs)


def config_of(model):
    mods = dict(model.named_modules())
    return {r: mods[r].out_channels for r in UNIT_ROOTS}


def materialize(base, move, cfg):
    """기준 가중치에서 단위별 채널 수가 cfg가 되도록 movement 순위가 낮은 채널부터 지운다."""
    m = copy.deepcopy(base)
    ranker = Ranker.build('mov_cum', base, move)
    dg = build_dg(m)
    name_of = {mm: n for n, mm in m.named_modules()}
    mods = dict(m.named_modules())
    for root in UNIT_ROOTS:
        mod = mods[root]
        k = mod.out_channels - cfg[root]
        if k <= 0:
            continue
        full = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=list(range(mod.out_channels)))
        idxs = torch.argsort(ranker.scores(m, full))[:k].tolist()
        group = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=idxs)
        ranker.remove(group, name_of)
        group.prune()
    return m


def probe(base, move, cfg, ctx, mode):
    return evaluate_probe(materialize(base, move, cfg), ctx, mode)  # (검증 NLL, 검증 정확도)


def better(a, b):
    """탐침 결과 비교: NLL이 낮을수록, 같으면 정확도가 높을수록 좋다."""
    return (a[0], -a[1]) < (b[0], -b[1])


def joint_greedy(base, move, targets, ctx, spatial):
    """두 제약(파라미터, MACs)을 함께 거는 탐욕 TRP. 매 반복 예산을 더 많이 넘은 자원을 줄인다.

    targets: [(파라미터 목표, MACs 목표), ...] 큰 것부터. 목표마다 그때의 구성을 돌려준다 (못 닿으면 None).
    """
    cur = copy.deepcopy(base)
    ranker = Ranker.build('mov_cum', base, move)
    out = []
    for pt, mt in targets:
        while True:
            p, mc = resources(cur)
            rp, rm = p / pt, mc / mt
            if rp <= 1.0001 and rm <= 1.0001:
                break
            kind, rc, tgt = ('params', p, pt) if rp >= rm else ('macs', mc, mt)
            need = rc - max(rc * (1 - ctx['step_frac']), tgt)
            cands = []
            for _ in range(6):
                cands = [c for c in (try_unit(cur, ranker, r, need, kind, spatial, ctx, 'bn') for r in UNIT_ROOTS) if c]
                if cands:
                    break
                need /= 2
            if not cands:
                return out + [None] * (len(targets) - len(out))
            best = min(cands, key=lambda c: (c['val_nll'], -c['val_acc']))
            dg = build_dg(cur)
            name_of = {mm: n for n, mm in cur.named_modules()}
            group = dg.get_pruning_group(dict(cur.named_modules())[best['root']], tp.prune_conv_out_channels,
                                         idxs=best['idxs'])
            ranker.remove(group, name_of)
            group.prune()
        out.append(config_of(cur))
    return out


def unit_costs(base, move, cfg, spatial):
    """현재 구성에서 단위마다 채널 하나가 차지하는 (파라미터, MACs)와 전체 자원."""
    m = materialize(base, move, cfg)
    dg = build_dg(m)
    name_of = {mm: n for n, mm in m.named_modules()}
    mods = dict(m.named_modules())
    costs = {}
    for root in UNIT_ROOTS:
        mod = mods[root]
        g = dg.get_pruning_group(mod, tp.prune_conv_out_channels, idxs=list(range(mod.out_channels)))
        costs[root] = (channel_cost(g, name_of, 'params', spatial), channel_cost(g, name_of, 'macs', spatial))
    return costs, resources(m)


def refine(base, move, cfg0, pt, mt, ctx, spatial, orig, rng):
    """한 단위를 늘리고 다른 단위를 줄이는 재배분을 BN 탐침으로 평가해, 나아질 때만 받아들인다."""
    cfg = dict(cfg0)
    cur = probe(base, move, cfg, ctx, 'bn')
    accepted, evaluated = 0, 0
    pairs = [(i, j) for i in UNIT_ROOTS for j in UNIT_ROOTS if i != j]
    for _ in range(ctx['rounds']):
        costs, (p, mc) = unit_costs(base, move, cfg, spatial)
        sp, sm = pt - p, mt - mc
        rng.shuffle(pairs)
        props = []
        for i, j in pairs[:ctx['moves']]:
            d = max(1, round(ctx['grow_frac'] * orig[i]))
            if cfg[i] + d > orig[i]:
                continue
            (cpi, cmi), (cpj, cmj) = costs[i], costs[j]
            s = max(0, math.ceil((d * cpi - sp) / cpj) if cpj > 0 else 0,
                    math.ceil((d * cmi - sm) / cmj) if cmj > 0 else 0)
            new = dict(cfg)
            new[i] += d
            new[j] -= s
            model = None
            for _ in range(4):  # 비용은 근사라서 실제로 재 보고, 넘으면 j를 한 채널씩 더 줄인다
                if new[j] < ctx['min_keep']:
                    break
                m = materialize(base, move, new)
                pp, mm = resources(m)
                if pp <= pt * 1.0001 and mm <= mt * 1.0001:
                    model = m
                    break
                new[j] -= 1
            if model is None:
                continue
            props.append((*evaluate_probe(model, ctx, 'bn'), new))
            evaluated += 1
        if not props:
            break
        best = min(props, key=lambda x: (x[0], -x[1]))
        if best[0] < cur[0] - ctx['eps']:
            cfg, cur = best[2], best[:2]
            accepted += 1
        else:
            break
    return cfg, accepted, evaluated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--tasks', default='auto')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.4, 0.6, 0.8, 0.9],
                    help='예산으로 쓸 균일 프루닝 비율 (그 모델의 파라미터와 MACs가 예산)')
    ap.add_argument('--probe-steps', type=int, default=56)
    ap.add_argument('--final-steps', type=int, default=840)
    ap.add_argument('--step-frac', type=float, default=0.2)
    ap.add_argument('--min-keep', type=int, default=4)
    ap.add_argument('--rounds', type=int, default=6, help='재배분 최대 반복 수')
    ap.add_argument('--moves', type=int, default=16, help='반복마다 평가할 재배분 후보 수')
    ap.add_argument('--grow-frac', type=float, default=0.1, help='한 번에 늘릴 채널 = 원래 너비의 이 비율')
    ap.add_argument('--eps', type=float, default=0.003, help='이만큼 이상 검증 NLL이 줄어야 재배분을 받아들임')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='joint')
    args = ap.parse_args()

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    names = auto_tasks(args.work_dir) if args.tasks == 'auto' else args.tasks.split(',')
    jobs = [(t, f, s) for t in names for f in args.folds for s in args.seeds]
    mine = jobs[args.shard::args.nshards]
    ratios = sorted(args.ratios)
    print(f'과제 {names}\n기준 모델 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개, 비율 {ratios}')

    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for n, (t, f, s) in enumerate(mine, 1):
        out_dir = runs / args.out_name / t / f'fold{f}_seed{s}'
        if all((out_dir / f'trp_joint_r{r}.json').exists() for r in ratios):
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
               'min_keep': args.min_keep, 'rounds': args.rounds, 'moves': args.moves, 'grow_frac': args.grow_frac,
               'eps': args.eps, 'base_stats': P.model_stats(base)}
        spatial = conv_spatial(base)
        orig = config_of(base)
        rng = random.Random(s)
        print(f'[{n}/{len(mine)}] {t} fold{f} seed{s}', flush=True)

        uniform = {r: {k: max(args.min_keep, int(orig[k] * (1 - r))) for k in UNIT_ROOTS} for r in ratios}
        targets = {r: resources(materialize(base, move, uniform[r])) for r in ratios}  # 균일 모델의 (파라미터, MACs)
        greedy = dict(zip(ratios, joint_greedy(base, move, [targets[r] for r in ratios], ctx, spatial)))
        print(f'  탐욕 탐색 끝 ({time.time() - t0:.0f}s)', flush=True)

        for r in ratios:
            out = out_dir / f'trp_joint_r{r}.json'
            if out.exists():
                continue
            tr = time.time()
            pt, mt = targets[r]
            common = {'task': t, 'fold': f, 'seed': s, 'ratio': r, 'target_params': pt, 'target_macs': mt,
                      'final_steps': args.final_steps}
            if greedy[r] is not None:
                finalize(materialize(base, move, greedy[r]), ctx, out_dir / f'joint_greedy_r{r}.json',
                         {**common, 'method': 'joint_greedy', 'budget': r, 'cfg': greedy[r]})
            starts = {'uniform': uniform[r]}
            if greedy[r] is not None:
                starts['greedy'] = greedy[r]
            scores = {k: probe(base, move, c, ctx, 'recover') for k, c in starts.items()}
            start = min(scores, key=lambda k: (scores[k][0], -scores[k][1]))
            cfg, accepted, evaluated = refine(base, move, starts[start], pt, mt, ctx, spatial, orig, rng)
            chosen = start
            if accepted:
                sc = probe(base, move, cfg, ctx, 'recover')
                if better(sc, scores[start]):
                    chosen = 'refined'
                else:
                    cfg = starts[start]
            finalize(materialize(base, move, cfg), ctx, out,
                     {**common, 'method': 'trp_joint', 'budget': r, 'start': start, 'chosen': chosen,
                      'accepted_moves': accepted, 'evaluated_moves': evaluated,
                      'start_scores': {k: list(v) for k, v in scores.items()}, 'cfg': cfg,
                      'search_sec': round(time.time() - tr, 1)})
        print(f'  완료 ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
