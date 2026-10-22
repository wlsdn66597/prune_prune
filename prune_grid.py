"""4·5단계: 기준 모델마다 (기준 × 비율)로 한 번에 자르고, BN 재보정 후 평가한다.

--recover-steps 0   : 4단계, 자른 직후 평가 (학습 없음)
--recover-steps 280 : 5단계, 회복 학습 후 평가 (짧은 예산은 56)

사용: python prune_grid.py --shard 0 --nshards 3
      python prune_grid.py --recover-steps 280 --ratios 0.6 0.8 --shard 0 --nshards 3
끝난 조합은 건너뛴다.
"""
import argparse
import copy
import time

import torch

import common as C
import prune_common as P


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--tasks', default='all', help="'all' 또는 쉼표로 구분한 과제 이름")
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--criteria', nargs='+', default=P.CRITERIA, choices=P.CRITERIA + P.EXTRA_CRITERIA)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
    ap.add_argument('--recover-steps', type=int, default=0)
    ap.add_argument('--taylor-batches', type=int, default=20)
    ap.add_argument('--ignore-conv1', action='store_true', help='첫 conv(와 layer1 잔차 채널)를 자르지 않는다')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--base-name', default='base', help='기준 모델 폴더 이름')
    ap.add_argument('--out-name', default='prune', help='결과 폴더 이름 (시험 실행은 다른 이름으로)')
    args = ap.parse_args()

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    names = list(tasks) if args.tasks == 'all' else args.tasks.split(',')
    unknown = [n for n in names if n not in tasks]
    if unknown:
        raise SystemExit(f'없는 과제 이름: {unknown}')
    jobs = [(t, f, s) for t in names for f in args.folds for s in args.seeds]
    mine = jobs[args.shard::args.nshards]
    print(f'기준 모델 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개, '
          f'모델마다 {len(args.criteria)}기준 × {len(args.ratios)}비율')

    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for n, (t, f, s) in enumerate(mine, 1):
        base_dir = runs / args.base_name / t / f'fold{f}_seed{s}'
        if not (base_dir / 'result.json').exists():
            print(f'기준 모델 없음, 건너뜀: {base_dir}')
            continue
        task = tasks[t]
        k = len(task['classes'])
        split = C.task_split(images, task, f)
        (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
        train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
        base = P.load_base(base_dir, k, args.device)
        move = P.load_movement(base_dir)
        base_stats = P.model_stats(base)
        t0 = time.time()
        for crit in args.criteria:
            for r in args.ratios:
                out = runs / args.out_name / t / f'fold{f}_seed{s}' / f'{crit}_r{r}_rec{args.recover_steps}.json'
                if out.exists():
                    continue
                C.set_seed(s)
                model = copy.deepcopy(base)
                if crit == 'taylor':
                    P.accumulate_grads(model, train_ds, args.device, args.taylor_batches, s, args.workers)
                P.prune(model, P.make_importance(crit, model, move), r, args.ignore_conv1)
                P.recalibrate_bn(model, cache, tr_idx)
                if args.recover_steps > 0:
                    P.recover(model, train_ds, args.recover_steps, args.device, args.workers, s)
                val_logits = C.predict_logits(model, cache, va_idx)
                test_logits = C.predict_logits(model, cache, te_idx)
                st = P.model_stats(model)
                info = {'task': t, 'fold': f, 'seed': s, 'criterion': crit, 'ratio': r,
                        'recover_steps': args.recover_steps, 'ignore_conv1': args.ignore_conv1,
                        'val': C.metrics(val_logits, va_y), 'test': C.metrics(test_logits, te_y),
                        'params': st['params'], 'macs': st['macs'],
                        'params_ratio': st['params'] / base_stats['params'],
                        'macs_ratio': st['macs'] / base_stats['macs'], 'channels': st['channels']}
                P.save_result(out, info, {'test_idx': te_idx, 'test_y': te_y, 'test_logits': test_logits.numpy()})
                del model
        print(f'[{n}/{len(mine)}] {t} fold{f} seed{s} 완료 ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
