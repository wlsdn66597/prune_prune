"""2단계: 과제 x fold 기준 모델을 학습하고 movement를 기록한다.

사용: python train_base.py                       # 전부 순서대로
      python train_base.py --shard 0 --nshards 3  # 프로세스 3개로 나눠서 (터미널마다 shard 0, 1, 2)
끝난 실험(result.json이 있는 폴더)은 건너뛰므로, 중간에 끊겨도 같은 명령으로 이어서 돌리면 된다.
"""
import argparse

import torch

import common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None, help='prepare_data.py 때와 데이터 경로가 달라졌을 때만')
    ap.add_argument('--tasks', default='all', help="'all' 또는 쉼표로 구분한 과제 이름 (예: pair_gl-pl,all6)")
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--total-steps', type=int, default=840, help='팀 레시피(15 epoch x 56 step)와 같은 업데이트 횟수')
    ap.add_argument('--eval-every', type=int, default=56)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out-name', default='base', help='결과 폴더 이름 (시험 실행은 다른 이름으로)')
    args = ap.parse_args()

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    if not tasks:
        raise SystemExit('tasks.json이 없습니다. make_folds.py를 먼저 실행하세요.')
    names = list(tasks) if args.tasks == 'all' else args.tasks.split(',')
    unknown = [n for n in names if n not in tasks]
    if unknown:
        raise SystemExit(f'없는 과제 이름: {unknown}\n가능한 이름: {list(tasks)}')
    jobs = [(t, f, s) for t in names for f in args.folds for s in args.seeds]
    mine = jobs[args.shard::args.nshards]
    print(f'전체 {len(jobs)}개 중 이 프로세스 담당 {len(mine)}개')

    cache = C.EvalCache(args.work_dir, args.device)
    for n, (t, f, s) in enumerate(mine, 1):
        out = C.runs_dir(args.work_dir) / args.out_name / t / f'fold{f}_seed{s}'
        if (out / 'result.json').exists():
            continue
        task = tasks[t]
        info = C.run_job(images, C.task_split(images, task, f), len(task['classes']), cache, out, seed=s,
                         total_steps=args.total_steps, eval_every=args.eval_every, device=args.device,
                         workers=args.workers, tag=f'{t} f{f} s{s}',
                         extra={'task': t, 'fold': f, 'classes': task['classes']})
        print(f"[{n}/{len(mine)}] {t} fold{f} seed{s}: test acc={info['test']['acc']:.4f} "
              f"nll={info['test']['nll']:.3f} best_step={info['best_step']} ({info['time_sec']}s)", flush=True)


if __name__ == '__main__':
    main()
