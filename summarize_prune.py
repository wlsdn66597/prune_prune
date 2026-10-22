"""4·5단계 결과 요약: 기준 × 비율별 test 정확도·NLL (과제·fold 평균)과 L2 대비 차이.

사용: python summarize_prune.py                    # 자른 직후 (rec0)
      python summarize_prune.py --recover-steps 280
"""
import argparse
import json

import pandas as pd

import common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--out-name', default='prune')
    ap.add_argument('--recover-steps', type=int, default=0)
    args = ap.parse_args()
    pd.set_option('display.width', 220)

    rows = []
    for f in (C.runs_dir(args.work_dir) / args.out_name).rglob(f'*_rec{args.recover_steps}.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append({'task': r['task'], 'fold': r['fold'], 'seed': r['seed'], 'crit': r['criterion'],
                     'ratio': r['ratio'], 'acc': r['test']['acc'], 'nll': r['test']['nll'],
                     'params_ratio': r['params_ratio'], 'macs_ratio': r['macs_ratio']})
    if not rows:
        raise SystemExit('결과가 없습니다.')
    df = pd.DataFrame(rows)
    df.to_csv(C.runs_dir(args.work_dir) / f'prune_results_rec{args.recover_steps}.csv', index=False)
    print(f'결과 {len(df)}개, 과제 {df.task.nunique()}개\n')

    print('== 남은 파라미터·연산량 비율 (비율별 평균) ==')
    print(df.groupby('ratio')[['params_ratio', 'macs_ratio']].mean().round(3).to_string())

    for metric in ('acc', 'nll'):
        print(f'\n== test {metric}: 기준 × 비율 (전체 과제·fold 평균) ==')
        print(df.pivot_table(index='crit', columns='ratio', values=metric, aggfunc='mean').round(4).to_string())

    # L2 대비 정확도 차이: 같은 과제·fold·비율끼리 짝지어 뺀 뒤 평균, 이긴 비율
    key = ['task', 'fold', 'seed', 'ratio']
    l2 = df[df.crit == 'l2'].set_index(key)['acc']
    d = df[df.crit != 'l2'].join(l2.rename('acc_l2'), on=key)
    d['diff'] = d['acc'] - d['acc_l2']
    print('\n== L2 대비 정확도 차이 (%p, 짝지은 평균) ==')
    print((d.pivot_table(index='crit', columns='ratio', values='diff', aggfunc='mean') * 100).round(2).to_string())
    print('\n== L2보다 높은 비율 (동점 제외) ==')
    d['win'] = (d['diff'] > 0).astype(float)
    d.loc[d['diff'] == 0, 'win'] = float('nan')
    print(d.pivot_table(index='crit', columns='ratio', values='win', aggfunc='mean').round(2).to_string())


if __name__ == '__main__':
    main()
