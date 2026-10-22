"""3단계: movement 근사 검증. 학습 없이 저장된 movement.pt만 읽는다.

체크포인트 근사 sum(w_best^2 - w_pre^2)와 학습 중 누적 movement(-grad * weight)의
층별 채널 순위 상관(Spearman)과, 60%를 잘랐을 때 잘리는 채널이 얼마나 겹치는지(Jaccard)를 본다.

사용: python check_movement.py
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import common as C

RATIO = 0.6
RANDOM_JACCARD = RATIO ** 2 / (2 * RATIO - RATIO ** 2)  # 무작위로 고른 두 60% 집합의 기대 Jaccard


def jaccard_pruned(a, b, ratio=RATIO):
    k = int(round(len(a) * ratio))
    sa, sb = set(np.argsort(a)[:k]), set(np.argsort(b)[:k])  # 점수가 낮은 채널부터 잘린다
    return len(sa & sb) / len(sa | sb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    args = ap.parse_args()

    rows = []
    for f in sorted((C.runs_dir(args.work_dir) / 'base').glob('*/*/movement.pt')):
        mv = torch.load(f, map_location='cpu', weights_only=False)
        b = mv['best_i']
        for n in mv['conv_names']:
            ckpt = mv['ckpt_out'][n].numpy()
            cum = mv['cum_out'][n][b].numpy()
            late = (mv['cum_out'][n][b] - mv['cum_out'][n][0]).numpy()
            rows.append({'task': f.parent.parent.name, 'run': f.parent.name, 'conv': n,
                         'stage': n.split('.')[0], 'channels': len(ckpt),
                         'rho_ckpt_cum': spearmanr(ckpt, cum).correlation,
                         'jac_ckpt_cum': jaccard_pruned(ckpt, cum),
                         'rho_cum_late': spearmanr(cum, late).correlation})
    if not rows:
        raise SystemExit('movement.pt가 없습니다. train_base.py를 먼저 실행하세요.')
    df = pd.DataFrame(rows)
    out = C.runs_dir(args.work_dir) / 'movement_check.csv'
    df.to_csv(out, index=False)

    cols = ['rho_ckpt_cum', 'jac_ckpt_cum', 'rho_cum_late']
    pd.set_option('display.width', 200)
    print(f'층 {len(df)}개 (실행 {df["run"].nunique() * df["task"].nunique()}개 이하)')
    print(f'참고: 무작위로 고른 두 {int(RATIO * 100)}% 집합의 기대 Jaccard = {RANDOM_JACCARD:.2f}\n')
    print('== 전체 평균 ==')
    print(df[cols].describe().loc[['mean', 'std', 'min', '50%']].round(3).to_string())
    print('\n== stage별 평균 ==')
    print(df.groupby('stage')[cols].mean().round(3).to_string())
    print('\n== 과제별 평균 ==')
    print(df.groupby('task')[cols].mean().round(3).sort_values('rho_ckpt_cum').to_string())
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
