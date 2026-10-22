"""결과 요약: work/runs의 result.json을 모아 표로 보여 주고 CSV로 저장한다.

사용: python summarize.py
"""
import argparse
import json

import pandas as pd

import common as C


def load(work, kind):
    rows = []
    for f in sorted((C.runs_dir(work) / kind).rglob('result.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'task': r['task'], 'fold': r.get('fold'), 'seed': r['seed'],
                     'test_acc': r['test']['acc'], 'test_f1': r['test']['macro_f1'],
                     'test_nll': r['test']['nll'], 'best_step': r['best_step'],
                     'n_train': r['n_train'], 'n_test': r['n_test'], 'sec': r['time_sec']})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    runs = C.runs_dir(args.work_dir)

    team = load(args.work_dir, 'team')
    if not team.empty:
        print('== 1단계: 팀 분할 재현 (노트북: acc 0.9140 ± 0.0132, macro-F1 0.9074 ± 0.0085) ==')
        print(team[['seed', 'test_acc', 'test_f1', 'test_nll', 'best_step', 'sec']].to_string(index=False))
        print(team[['test_acc', 'test_f1']].agg(['mean', 'std']).round(4).to_string())

    base = load(args.work_dir, 'base')
    if not base.empty:
        g = base.groupby('task').agg(runs=('fold', 'size'), acc=('test_acc', 'mean'), acc_sd=('test_acc', 'std'),
                                     f1=('test_f1', 'mean'), nll=('test_nll', 'mean'),
                                     best_step=('best_step', 'mean'), sec=('sec', 'mean'))
        sim_file = C.meta_dir(args.work_dir) / 'similarity.csv'
        if sim_file.exists():
            g = g.join(pd.read_csv(sim_file).set_index('task')[['k', 'silhouette']]).sort_values(['k', 'silhouette'])
        print('\n== 2단계: 과제별 기준 모델 (fold 평균) ==')
        print(g.round(4).to_string())
        base.to_csv(runs / 'base_results.csv', index=False)
        g.to_csv(runs / 'base_summary.csv')
        print(f'\n저장: {runs / "base_results.csv"}, {runs / "base_summary.csv"}')


if __name__ == '__main__':
    main()
