"""6단계: stage별 민감도. 한 번에 한 stage(layer1~4)의 블록 내부 채널만 자른다.

블록 내부 채널은 각 BasicBlock의 conv1 출력 = conv2 입력 채널이다. 잔차로 묶인 채널은 자르지 않는다.
과제는 기본으로 6클래스, 가장 어려운/쉬운 쌍, 가장 어려운/쉬운 삼중 5개를 similarity.csv에서 고른다.

사용: python stage_sensitivity.py --shard 0 --nshards 3            # 자른 직후 (BN 재보정 후)
      python stage_sensitivity.py --recover-steps 840 --shard 0 --nshards 3
      python stage_sensitivity.py --summary [--recover-steps 840]   # 결과 표
"""
import argparse
import copy
import json

import pandas as pd
import torch

import common as C
import prune_common as P

STAGES = ['layer1', 'layer2', 'layer3', 'layer4']


def auto_tasks(work):
    sim = pd.read_csv(C.meta_dir(work) / 'similarity.csv')
    names = ['all6']
    for k in (2, 3):
        s = sim[sim.k == k].sort_values('silhouette')
        names += [s.task.iloc[0], s.task.iloc[-1]]  # 가장 어려운 것, 가장 쉬운 것
    return names


def summary(args):
    rows = []
    for f in (C.runs_dir(args.work_dir) / args.out_name).rglob(f'*_rec{args.recover_steps}.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'task': r['task'], 'crit': r['criterion'], 'stage': r['stage'], 'ratio': r['ratio'],
                     'acc': r['test']['acc'], 'params_ratio': r['params_ratio'], 'macs_ratio': r['macs_ratio']})
    if not rows:
        raise SystemExit('결과가 없습니다.')
    df = pd.DataFrame(rows)
    base = pd.read_csv(C.runs_dir(args.work_dir) / 'base_summary.csv').set_index('task')['acc']
    df['drop'] = (df['task'].map(base) - df['acc']) * 100
    pd.set_option('display.width', 220)
    print('== stage별로 잘랐을 때 남는 파라미터·연산량 비율 ==')
    print(df.groupby(['stage', 'ratio'])[['params_ratio', 'macs_ratio']].mean().unstack('ratio').round(3).to_string())
    for crit in sorted(df.crit.unique()):
        print(f'\n== {crit}: 원본 대비 정확도 하락(%p), 과제 × stage, fold 평균 ==')
        t = df[df.crit == crit].pivot_table(index=['task', 'stage'], columns='ratio', values='drop', aggfunc='mean')
        print(t.round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--tasks', default='auto', help="'auto' 또는 쉼표로 구분한 과제 이름")
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--criteria', nargs='+', default=['l2', 'mov_cum'], choices=P.CRITERIA + P.EXTRA_CRITERIA)
    ap.add_argument('--stages', nargs='+', default=STAGES, choices=STAGES)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.5, 0.8, 0.9])
    ap.add_argument('--recover-steps', type=int, default=0)
    ap.add_argument('--taylor-batches', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='stage')
    ap.add_argument('--summary', action='store_true', help='실험 없이 결과 표만 출력')
    args = ap.parse_args()
    if args.summary:
        return summary(args)

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    names = auto_tasks(args.work_dir) if args.tasks == 'auto' else args.tasks.split(',')
    jobs = [(t, f, s) for t in names for f in args.folds for s in args.seeds]
    mine = jobs[args.shard::args.nshards]
    print(f'과제 {names}\n기준 모델 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개')

    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for n, (t, f, s) in enumerate(mine, 1):
        base_dir = runs / 'base' / t / f'fold{f}_seed{s}'
        task = tasks[t]
        split = C.task_split(images, task, f)
        (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
        train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
        base = P.load_base(base_dir, len(task['classes']), args.device)
        move = P.load_movement(base_dir)
        base_stats = P.model_stats(base)
        for crit in args.criteria:
            for stage in args.stages:
                for r in args.ratios:
                    out = runs / args.out_name / t / f'fold{f}_seed{s}' / f'{crit}_{stage}_r{r}_rec{args.recover_steps}.json'
                    if out.exists():
                        continue
                    C.set_seed(s)
                    model = copy.deepcopy(base)
                    if crit == 'taylor':
                        P.accumulate_grads(model, train_ds, args.device, args.taylor_batches, s, args.workers)
                    ratio_dict = {blk.conv1: r for blk in getattr(model, stage)}
                    P.prune(model, P.make_importance(crit, model, move), 0.0, ratio_dict=ratio_dict)
                    P.recalibrate_bn(model, cache, tr_idx)
                    if args.recover_steps > 0:
                        P.recover(model, train_ds, args.recover_steps, args.device, args.workers, s)
                    test_logits = C.predict_logits(model, cache, te_idx)
                    st = P.model_stats(model)
                    info = {'task': t, 'fold': f, 'seed': s, 'criterion': crit, 'stage': stage, 'ratio': r,
                            'recover_steps': args.recover_steps,
                            'val': C.metrics(C.predict_logits(model, cache, va_idx), va_y),
                            'test': C.metrics(test_logits, te_y),
                            'params': st['params'], 'macs': st['macs'],
                            'params_ratio': st['params'] / base_stats['params'],
                            'macs_ratio': st['macs'] / base_stats['macs'], 'channels': st['channels']}
                    P.save_result(out, info, {'test_idx': te_idx, 'test_y': te_y,
                                              'test_logits': test_logits.numpy()})
                    del model
        print(f'[{n}/{len(mine)}] {t} fold{f} seed{s} 완료', flush=True)


if __name__ == '__main__':
    main()
