"""기성 소형 모델 비교군: MobileNetV3-Small/Large, MobileNetV2를 같은 전처리·분할·학습량(840 step)으로 파인튜닝한다.

'ResNet-18을 줄이는 대신 처음부터 작은 모델을 쓰면 되지 않나?'에 답하기 위한 비교군이다.
비교군이 불리하지 않도록 학습률을 1e-4와 1e-3 둘 다 학습하고, summarize_mobile.py에서 검증 정확도로 고른다.
(ResNet-18은 팀 레시피인 1e-4 그대로라, 비교군 쪽이 조금 유리한 조건이다.)
가중치는 torchvision이 권장하는 ImageNet 사전학습 가중치(DEFAULT)를 쓰고, 마지막 Linear만 클래스 수에 맞게 바꾼다.

사용: python train_mobile.py --shard 0 --nshards 3
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import common as C
import prune_common as P

ARCHS = {'mnv3s': 'mobilenet_v3_small', 'mnv3l': 'mobilenet_v3_large', 'mnv2': 'mobilenet_v2',
         'effb0': 'efficientnet_b0'}


def build(arch, num_classes):
    name = ARCHS[arch]
    weights = models.get_model_weights(name).DEFAULT
    m = models.get_model(name, weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m, str(weights)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--archs', nargs='+', default=['mnv3s', 'mnv2', 'mnv3l'], choices=list(ARCHS))
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-4, 1e-3])
    ap.add_argument('--tasks', default='all', help="'all', 'auto'(5과제) 또는 쉼표로 구분")
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--total-steps', type=int, default=840)
    ap.add_argument('--eval-every', type=int, default=56)
    ap.add_argument('--save-model', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='mobile')
    args = ap.parse_args()

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    if args.tasks == 'all':
        names = list(tasks)
    elif args.tasks == 'auto':
        from stage_sensitivity import auto_tasks
        names = auto_tasks(args.work_dir)
    else:
        names = args.tasks.split(',')
    jobs = [(t, f, s, a, lr) for t in names for f in args.folds for s in args.seeds
            for a in args.archs for lr in args.lrs]
    mine = jobs[args.shard::args.nshards]
    print(f'전체 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개 (모델 {args.archs}, 학습률 {args.lrs})')

    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for n, (t, f, s, arch, lr) in enumerate(mine, 1):
        out = runs / args.out_name / t / f'fold{f}_seed{s}' / f'{arch}_lr{lr:g}.json'
        if out.exists():
            continue
        t0 = time.time()
        task = tasks[t]
        split = C.task_split(images, task, f)
        (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
        C.set_seed(s)
        model, wname = build(arch, len(task['classes']))
        train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
        hist, _ = C.train_model(model, train_ds, cache, va_idx, va_y, seed=s, total_steps=args.total_steps,
                                eval_every=args.eval_every, device=args.device, workers=args.workers,
                                tag=f'{arch} lr{lr:g} {t} f{f}', lr=lr)
        val_logits = C.predict_logits(model, cache, va_idx)
        test_logits = C.predict_logits(model, cache, te_idx)
        st = P.model_stats(model)
        info = {'task': t, 'fold': f, 'seed': s, 'arch': arch, 'weights': wname, 'lr': lr,
                'total_steps': args.total_steps, 'best_val_acc': max(hist['val_acc']),
                'best_step': hist['step'][int(np.argmax(hist['val_acc']))],
                'val': C.metrics(val_logits, va_y), 'test': C.metrics(test_logits, te_y),
                'params': st['params'], 'macs': st['macs'], 'time_sec': round(time.time() - t0, 1)}
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.save_model:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out.with_suffix('.pt'))
        np.savez_compressed(out.with_suffix('.npz'), test_idx=te_idx, test_y=te_y, test_logits=test_logits.numpy())
        out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"[{n}/{len(mine)}] {arch} lr{lr:g} {t} f{f}: test acc={info['test']['acc']:.4f} "
              f"params={st['params'] / 1e6:.2f}M macs={st['macs'] / 1e9:.3f}G ({info['time_sec']}s)", flush=True)


if __name__ == '__main__':
    main()
