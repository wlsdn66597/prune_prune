"""Matched Taylor ablation: class-balanced C versus pooled-gradient E.

Both arms use exactly the same sampled training images, augmentations, forward/
backward passes, base model, pruning ratio and 2800-step recovery. C applies abs
and per-convolution standardization before averaging class scores (the existing
balanced formula); E pools signed gradients before abs and standardization.
E still uses the same class-stratified probe sample to avoid a sampling confound.
It is a pooled Taylor control, not global channel pruning.

Historical A/B/C results are reported separately, not overwritten. The primary
comparison is freshly matched C - E. Threshold fixed before this run: mean
accuracy gain >= 1 pp and wins in >= 4/5 seeds is supporting evidence, not proof
of a universal mechanism. A smaller effect is inconclusive, not equivalence.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import common as C
import prune_common as P
from plan_eval import REC_STEPS
from team_protect import N_BATCH, protect_scores
from team_weighted import class_weights

VERSION = 'matched_taylor_v1'
ARMS = ['balanced_matched', 'global']
NAMES = {'mov': 'A movement (existing)', 'protect': 'B risk-weighted (existing)',
         'balanced': 'C balanced (existing)',
         'balanced_matched': 'C balanced (matched)', 'global': 'E pooled Taylor (matched)'}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def metrics(logits, y):
    result = C.metrics(logits, y)
    pred, labels = logits.argmax(1).numpy(), np.asarray(y)
    result['recall'] = {c: float((pred[labels == i] == i).mean())
                        for i, c in enumerate(C.CLASSES6)}
    return result


def matched_taylor(model, dataset, labels, args, seed):
    """One shared probe; retain signed channel sums until the pooling decision."""
    C.set_seed(seed)
    model.eval()
    convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    pooled_o = {n: torch.zeros(m.out_channels) for n, m in convs.items()}
    pooled_i = {n: torch.zeros(m.in_channels) for n, m in convs.items()}
    classes, counts = {}, {}
    for c in np.unique(labels):
        loader = DataLoader(Subset(dataset, np.flatnonzero(labels == c).tolist()),
                            C.BATCH, shuffle=True, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(seed))
        model.zero_grad(set_to_none=True)
        batches = examples = 0
        for k, (x, y) in enumerate(loader):
            if k >= N_BATCH:
                break
            F.cross_entropy(model(x.to(args.device)), y.to(args.device)).backward()
            batches += 1
            examples += len(y)
        if batches == 0:
            raise RuntimeError(f'No probe samples for class {c}')
        out, inp = {}, {}
        for n, m in convs.items():
            if m.weight.grad is None:
                raise RuntimeError(f'Missing gradient: {n}')
            gw = (m.weight.grad * m.weight).detach()
            signed_o = gw.sum(dim=(1, 2, 3)).cpu()
            signed_i = gw.sum(dim=(0, 2, 3)).cpu()
            out[n], inp[n] = signed_o.abs(), signed_i.abs()
            pooled_o[n] += signed_o
            pooled_i[n] += signed_i
        classes[int(c)] = (out, inp)
        counts[int(c)] = {'batches': batches, 'images': examples}
        del loader
    model.zero_grad(set_to_none=True)
    pooled = {0: ({n: x.abs() for n, x in pooled_o.items()},
                  {n: x.abs() for n, x in pooled_i.items()})}
    return classes, pooled, counts


def preflight(args):
    work = Path(args.work_dir)
    required = [work / 'meta' / n for n in
                ['images.csv', 'config.json', 'eval224_uint8.pt', 'feat_pretrained.npy']]
    required += [C.runs_dir(work) / 'team' / f'seed{s}' / n
                 for s in args.seeds for n in ['model.pt', 'movement.pt']]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit('Missing inputs (no replacement baselines will be trained):\n' + '\n'.join(missing))
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; refusing to start the full run on CPU.')
    images, _ = C.load_meta(args.work_dir, args.data_root)
    missing_images = [p for p in images['path'] if not Path(p).is_file()]
    if missing_images:
        raise SystemExit(f'Missing {len(missing_images)} images; first: {missing_images[0]}')
    print(f'Preflight OK: seeds={args.seeds}, ratio={args.ratio}, recovery={args.steps}, '
          f'device={args.device}, torch={torch.__version__}', flush=True)


def train(args):
    preflight(args)
    root = Path(args.work_dir) / args.out_name
    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    tr_idx, tr_y = split['train']
    dataset = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    cache = C.EvalCache(args.work_dir, args.device)
    protocol = {'version': VERSION, 'seeds': args.seeds, 'ratio': args.ratio,
                'steps': args.steps, 'workers': args.workers, 'probe_batches_per_class': N_BATCH,
                'batch_size': C.BATCH, 'arms': ARMS, 'primary': 'balanced_matched - global',
                'supporting_threshold_pp': 1.0, 'supporting_win_fraction': 0.8,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'note': 'Shared class-stratified probe; batch-mean CE as in existing class_taylor. '
                        'Existing results are descriptive references only.'}
    manifest = root / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != protocol:
        raise SystemExit('Protocol differs from existing output. Use a fresh --out-name.')
    dump(manifest, protocol)
    for seed in args.seeds:
        paths = {a: root / a / f'seed{seed}' / f'mov_r{args.ratio}.json' for a in ARMS}
        if all(p.exists() for p in paths.values()):
            print(f'seed{seed}: both arms already complete; skipping', flush=True)
            continue
        base_dir = C.runs_dir(args.work_dir) / 'team' / f'seed{seed}'
        base = P.load_base(base_dir, len(C.CLASSES6), args.device)
        move = P.load_movement(base_dir)
        t0 = time.time()
        class_scores, pooled_scores, counts = matched_taylor(base, dataset, np.asarray(tr_y), args, seed)
        scores = {}
        for arm, probe in [('balanced_matched', class_scores), ('global', pooled_scores)]:
            _, out, inp = protect_scores(move, probe, np.ones(len(probe)))
            scores[arm] = (out, inp)
        probe_sec = time.time() - t0
        for arm in ARMS:
            path = paths[arm]
            if path.exists():
                continue
            started = time.time()
            C.set_seed(seed)
            model = copy.deepcopy(base)
            P.prune(model, P.ChannelScoreImportance(model, *scores[arm]), args.ratio)
            stats = P.model_stats(model)
            P.recalibrate_bn(model, cache, tr_idx)
            C.set_seed(seed)  # identical recovery batches/augmentations in the two arms
            P.recover(model, dataset, args.steps, args.device, args.workers, seed)
            val_idx, val_y = split['val']
            test_idx, test_y = split['test']
            info = {'version': VERSION, 'seed': seed, 'arm': arm, 'ratio': args.ratio,
                    'steps': args.steps, 'probe_counts': counts, 'shared_probe_sec': probe_sec,
                    'stats': stats, 'val': metrics(C.predict_logits(model, cache, val_idx), val_y),
                    'test': metrics(C.predict_logits(model, cache, test_idx), test_y),
                    'train_eval_sec': round(time.time() - started, 1)}
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu(), path.with_suffix('.pt'))
            dump(path, info)  # completion marker written last
            print(f"seed{seed} {arm}: test={100 * info['test']['acc']:.2f}%, "
                  f"params={stats['params']}, MACs={stats['macs']}, "
                  f"time={info['train_eval_sec']}s", flush=True)
            del model
        del base, move, scores, class_scores, pooled_scores
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()


def judge(args):
    root = Path(args.work_dir) / args.out_name
    rows, budgets = [], {}
    for seed in args.seeds:
        for arm in ARMS:
            path = root / arm / f'seed{seed}' / f'mov_r{args.ratio}.json'
            if not path.is_file():
                raise SystemExit(f'Incomplete primary comparison: {path}')
            info = json.loads(path.read_text(encoding='utf-8'))
            if info['steps'] != args.steps or info['version'] != VERSION:
                raise SystemExit(f'Incompatible result: {path}')
            budgets[seed, arm] = info['stats']
            rows.append(make_row(seed, arm, info['test']))
        if budgets[seed, ARMS[0]] != budgets[seed, ARMS[1]]:
            raise SystemExit(f'Budget/structure mismatch in seed{seed}; no verdict.')

    # Evaluate existing checkpoints only for reference; never write to their folders.
    images, _ = C.load_meta(args.work_dir, args.data_root)
    test_idx, test_y = C.team_split(images)['test']
    cache = None
    for seed in args.seeds:
        for arm in ['mov', 'protect', 'balanced']:
            path = Path(args.work_dir) / args.reference_name / arm / f'seed{seed}' / f'mov_r{args.ratio}.pt'
            if not path.is_file():
                print(f'Reference unavailable (not used for primary verdict): {path}')
                continue
            if cache is None:
                cache = C.EvalCache(args.work_dir, args.device)
            model = torch.load(path, map_location=args.device, weights_only=False).eval()
            rows.append(make_row(seed, arm, metrics(C.predict_logits(model, cache, test_idx), test_y)))
            del model
    frame = pd.DataFrame(rows)
    cols = ['accuracy', 'macroF1', *C.CLASSES6]
    print('\n=== Matched C/E Taylor ablation (test %, percentage points) ===')
    print(frame.groupby('arm')[cols].mean().rename(index=NAMES).round(2).to_string())
    print('\nAvailable seed count per arm: ' + str(frame.groupby('arm').size().to_dict()))
    paired = frame[frame.arm.isin(ARMS)].pivot(index='seed', columns='arm', values=cols)
    delta = paired.xs('balanced_matched', axis=1, level=1) - paired.xs('global', axis=1, level=1)
    print('\nPaired C(matched) - E by seed:')
    print(delta.round(2).to_string())
    for col in cols:
        x = delta[col]
        print(f'{col}: mean={x.mean():+.2f} pp, SD={x.std(ddof=1):.2f}, '
              f'win/tie/loss={int((x > 1e-6).sum())}/{int((x.abs() <= 1e-6).sum())}/{int((x < -1e-6).sum())}')
    sil, _ = class_weights(args.work_dir, args.data_root)
    weak = C.CLASSES6[int(np.argmin(sil))]
    print(f'Pretrained lowest-silhouette class: {weak}; C-E recall={delta[weak].mean():+.2f} pp')
    gain = float(delta.accuracy.mean())
    wins = int((delta.accuracy > 1e-6).sum())
    if len(args.seeds) < 5:
        verdict = 'SMOKE/PARTIAL ONLY: at least five seeds required for the planned verdict.'
    elif gain >= 1.0 and wins / len(args.seeds) >= 0.8:
        verdict = 'Supporting evidence for class-wise aggregation beyond pooled Taylor; confirm on other splits/ratios.'
    else:
        verdict = 'Planned support criterion not met; this does NOT establish equivalence or a Taylor-only cause.'
    print('\nVerdict: ' + verdict)
    print('This C/E comparison does not establish a benefit of silhouette risk weighting.')
    frame.to_csv(root / 'per_seed_metrics.csv', index=False)
    delta.to_csv(root / 'paired_C_minus_E.csv')
    dump(root / 'summary.json', {'primary': 'C(matched) - E', 'seeds': args.seeds,
         'accuracy_delta_pp': gain, 'wins': wins, 'verdict': verdict,
         'mean_delta_pp': {k: float(v) for k, v in delta.mean().items()}})


def make_row(seed, arm, result):
    return {'seed': seed, 'arm': arm, 'accuracy': 100 * result['acc'],
            'macroF1': 100 * result['macro_f1'],
            **{c: 100 * result['recall'][c] for c in C.CLASSES6}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['check', 'train', 'judge'])
    parser.add_argument('--work-dir', default='work')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--out-name', default='team_taylor_control')
    parser.add_argument('--reference-name', default='team_protect2')
    parser.add_argument('--seeds', type=int, nargs='+', default=[6, 7, 8, 9, 10])
    parser.add_argument('--ratio', type=float, default=0.4)
    parser.add_argument('--steps', type=int, default=REC_STEPS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not 0 < args.ratio < 1 or args.steps <= 0:
        parser.error('Require unique seeds, 0 < ratio < 1, and positive recovery steps.')
    if Path(args.out_name).is_absolute() or '..' in Path(args.out_name).parts or args.out_name == args.reference_name:
        parser.error('Use a separate relative output name; reference results must not be overwritten.')
    pd.set_option('display.width', 200)
    {'check': preflight, 'train': train, 'judge': judge}[args.step](args)


if __name__ == '__main__':
    main()
