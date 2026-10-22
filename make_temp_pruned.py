"""양자화 담당용 임시 프루닝 모델: 팀 분할 기준 모델을 L2로 자르고 짧게 회복 학습한다.

최종 기준이 정해지면 교체될 임시 모델이다. 구조가 바뀌어서 모델 전체를 저장한다.
불러오기: model = torch.load('l2_r0.4.pt', weights_only=False)

사용: python make_temp_pruned.py
"""
import argparse
import json
import os

import torch

import common as C
import prune_common as P


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--seed', type=int, default=0, help='팀 분할 기준 모델의 seed')
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.4, 0.6])
    ap.add_argument('--recover-steps', type=int, default=280, help='약 5 epoch')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    (tr_idx, tr_y), _, (te_idx, te_y) = split['train'], split['val'], split['test']
    train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    cache = C.EvalCache(args.work_dir, args.device)
    base_dir = C.runs_dir(args.work_dir) / 'team' / f'seed{args.seed}'
    out_dir = C.runs_dir(args.work_dir).parent / 'temp_pruned'
    out_dir.mkdir(parents=True, exist_ok=True)

    base = P.load_base(base_dir, len(C.CLASSES6), args.device)
    acc0 = C.metrics(C.predict_logits(base, cache, te_idx), te_y)['acc']
    st0 = P.model_stats(base)
    print(f'원본: test acc={acc0:.4f}, params={st0["params"] / 1e6:.2f}M, MACs={st0["macs"] / 1e9:.2f}G')

    for r in args.ratios:
        C.set_seed(args.seed)
        model = P.prune(P.load_base(base_dir, len(C.CLASSES6), args.device), P.make_importance('l2', None, None), r)
        P.recalibrate_bn(model, cache, tr_idx)
        acc_pruned = C.metrics(C.predict_logits(model, cache, te_idx), te_y)['acc']
        P.recover(model, train_ds, args.recover_steps, args.device, args.workers, args.seed)
        m = C.metrics(C.predict_logits(model, cache, te_idx), te_y)
        st = P.model_stats(model)
        path = out_dir / f'l2_r{r}.pt'
        torch.save(model.cpu(), path)
        info = {'criterion': 'l2', 'ratio': r, 'base': str(base_dir), 'recover_steps': args.recover_steps,
                'test_acc_base': acc0, 'test_acc_after_prune': acc_pruned, 'test': m,
                'params': st['params'], 'macs': st['macs'], 'file_mb': round(os.path.getsize(path) / 1e6, 2),
                'channels': st['channels'], 'split': 'team_split.json (compare_fc.ipynb와 같은 분할)'}
        path.with_suffix('.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'비율 {r}: 자른 직후 {acc_pruned:.4f} -> 회복 후 {m["acc"]:.4f}, '
              f'params={st["params"] / 1e6:.2f}M, MACs={st["macs"] / 1e9:.2f}G, 파일 {info["file_mb"]}MB -> {path}')


if __name__ == '__main__':
    main()
