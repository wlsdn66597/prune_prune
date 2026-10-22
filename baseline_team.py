"""1단계: compare_fc.ipynb의 B 모델(fc6)을 같은 팀 분할에서 재현한다 (seed 0, 1, 2).

사용: python baseline_team.py
"""
import argparse
import json
import math

import numpy as np
import torch

import common as C

NOTEBOOK = {'acc': (0.9140, 0.0132), 'macro_f1': (0.9074, 0.0085)}  # compare_fc.ipynb의 B 모델 결과


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None, help='prepare_data.py 때와 데이터 경로가 달라졌을 때만')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out-name', default='team', help='결과 폴더 이름 (시험 실행은 다른 이름으로)')
    args = ap.parse_args()

    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    (C.meta_dir(args.work_dir) / 'team_split.json').write_text(
        json.dumps({k: v[0].tolist() for k, v in split.items()}), encoding='utf-8')
    print('train/val/test:', *(len(v[0]) for v in split.values()), '(노트북: 1768 379 380)')

    steps_per_epoch = math.ceil(len(split['train'][0]) / C.BATCH)  # 1768장이면 56
    cache = C.EvalCache(args.work_dir, args.device)
    results = []
    for seed in args.seeds:
        out = C.runs_dir(args.work_dir) / args.out_name / f'seed{seed}'
        if (out / 'result.json').exists():
            print(f'이미 있음, 건너뜀: {out}')
        else:
            C.run_job(images, split, len(C.CLASSES6), cache, out, seed=seed,
                      total_steps=args.epochs * steps_per_epoch, eval_every=steps_per_epoch,
                      device=args.device, workers=args.workers, tag=f'team seed{seed}',
                      extra={'task': 'team', 'classes': C.CLASSES6})
        results.append(json.loads((out / 'result.json').read_text(encoding='utf-8')))

    for key in ('acc', 'macro_f1'):
        v = np.array([r['test'][key] for r in results])
        sd = v.std(ddof=1) if len(v) > 1 else float('nan')
        m, s = NOTEBOOK[key]
        print(f'test {key}: {v.mean():.4f} ± {sd:.4f}   (노트북 {m:.4f} ± {s:.4f})')


if __name__ == '__main__':
    main()
