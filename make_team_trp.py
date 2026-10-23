"""팀 전달용 TRP 모델: 팀 분할(compare_fc.ipynb와 같은 분할)의 기준 모델을 trp_bn으로 예산까지 줄인다.

연구 실험에서 trp_bn이 trp와 성능이 같고 탐색이 더 빨라서 기본값으로 쓴다.
전달용이라 최종 회복은 1680 step으로 연구 실험(840)보다 길게 한다.
구조가 바뀐 모델이라 통째로 저장한다. 불러오기: torch.load(경로, weights_only=False)

사용: python make_team_trp.py
"""
import argparse
import copy
import json
import os
import time

import torch

import common as C
import prune_common as P
from trp_search import METHODS, Ranker, finalize, probe_batches, trp_search


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--seed', type=int, default=0, help='팀 분할 기준 모델의 seed')
    ap.add_argument('--method', choices=['trp_bn', 'trp'], default='trp_bn')
    ap.add_argument('--budgets', type=float, nargs='+', default=[0.16, 0.04], help='남길 파라미터 비율')
    ap.add_argument('--resource', choices=['params', 'macs'], default='params')
    ap.add_argument('--probe-steps', type=int, default=56)
    ap.add_argument('--final-steps', type=int, default=1680)
    ap.add_argument('--step-frac', type=float, default=0.2)
    ap.add_argument('--min-keep', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
    base_dir = C.runs_dir(args.work_dir) / 'team' / f'seed{args.seed}'
    base = P.load_base(base_dir, len(C.CLASSES6), args.device)
    move = P.load_movement(base_dir)
    base_acc = json.loads((base_dir / 'result.json').read_text(encoding='utf-8'))['test']['acc']
    train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    bx, by = probe_batches(train_ds, args.probe_steps, args.seed, args.workers, args.device)
    ctx = {'cache': C.EvalCache(args.work_dir, args.device), 'tr_idx': tr_idx, 'va_idx': va_idx, 'va_y': va_y,
           'te_idx': te_idx, 'te_y': te_y, 'train_ds': train_ds, 'bx': bx, 'by': by, 'seed': args.seed,
           'device': args.device, 'workers': args.workers, 'probe_steps': args.probe_steps,
           'final_steps': args.final_steps, 'step_frac': args.step_frac, 'min_keep': args.min_keep,
           'base_stats': P.model_stats(base)}
    out_dir = C.runs_dir(args.work_dir).parent / 'team_trp'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'원본 (팀 분할 seed{args.seed}): test acc={base_acc:.4f}, '
          f"params={ctx['base_stats']['params'] / 1e6:.2f}M, MACs={ctx['base_stats']['macs'] / 1e9:.2f}G")

    t0 = time.time()
    rank, mode = METHODS[args.method]

    def on_budget(b, cur, trace, n_probes):
        model = copy.deepcopy(cur)
        out = out_dir / f'{args.method}_b{b}.json'
        finalize(model, ctx, out, {'task': 'team', 'seed': args.seed, 'method': args.method, 'budget': b,
                                   'resource': args.resource, 'final_steps': args.final_steps,
                                   'n_probes': n_probes, 'search_sec': round(time.time() - t0, 1),
                                   'test_acc_base': base_acc, 'split': 'team_split.json'})
        path = out.with_suffix('.pt')
        torch.save(model.cpu(), path)
        info = json.loads(out.read_text(encoding='utf-8'))
        info['file_mb'] = round(os.path.getsize(path) / 1e6, 2)
        out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"예산 {b}: test acc={info['test']['acc']:.4f} (원본 {base_acc:.4f}), "
              f"params={info['params'] / 1e6:.2f}M, MACs={info['macs'] / 1e9:.3f}G, 파일 {info['file_mb']}MB -> {path}",
              flush=True)

    trace, _ = trp_search(base, Ranker.build(rank, base, move), args.resource, args.budgets, mode, ctx, on_budget)
    P.save_result(out_dir / f'{args.method}_trace.json', {'trace': trace})
    print(f'끝 ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
