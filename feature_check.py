"""3판 1단계 (학습 없음): '가장 약한 클래스의 실루엣'이 평균 실루엣보다 과제군 안에서 정확도를 더 잘 예측하나.

팀 분할 6분류에서 계획이 두 번 2~3%p 낙관적이었고, 자를 때 metal·glass가 먼저 무너졌다. 평균 실루엣은 한 클래스의
붕괴를 못 잡으므로 척도를 바꿔 본다. 과제군은 2판과 같다 (all6을 뺀 46과제, 기존 과제 fold 0~4, 새 과제 fold 0·1).
과제 하나씩 빼고 비율별 정확도를 그 척도 하나로 회귀해 예측하고(plan_diag의 곡선), 과제별(fold 평균) 실제와 비교한다.
  mean     : 실루엣 (지금까지 쓴 척도)
  minclass : 클래스별 평균 실루엣의 최솟값 (가장 약한 클래스, 3판 후보)
  minpair  : 클래스 쌍마다 계산한 실루엣의 최솟값 (가장 헷갈리는 쌍, 참고)
판정 (결과를 보기 전에 정함): minclass의 과제별 평균 절대 오차가 mean보다 10% 이상 작고,
  과제별 오차의 짝 비교 Wilcoxon p<0.05이면 통과. 통과하면 work/feature_check_pass를 만든다 (run_team_v3.sh가 본다).
5클래스 과제(6분류에 가장 가까움)의 부호 있는 오차도 같이 낸다 (양수면 낙관적).

사용: python feature_check.py > work/feature_check.txt
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import common as C
from plan_diag import MAIN, fit
from plan_eval import GRID, task_feature, train_silhouettes
from team_plan import load_family

KINDS = ['mean', 'minclass', 'minpair']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    runs = C.runs_dir(args.work_dir)
    flag = runs.parent / 'feature_check_pass'
    flag.unlink(missing_ok=True)

    V = load_family(runs, ext=True)
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    base = V.groupby(level='task').mean()
    images, _ = C.load_meta(args.work_dir, args.data_root)
    idx, y = C.team_split(images)['train']
    feats = np.load(C.meta_dir(args.work_dir) / 'feat_pretrained.npy')
    print(f'=== 3판 1단계: 척도별 과제군 안 예측 (과제 {len(base)}개, 조건 {len(V)}개, 과제 하나씩 빼고) ===')

    err, val = {}, {}
    for kind in KINDS:
        tt = base.copy()
        tt['sil'] = train_silhouettes(args.work_dir, args.data_root, list(V.index), kind).groupby(level='task').mean()
        tt['k'] = sim.k
        val[kind] = tt.sil
        err[kind] = pd.DataFrame({t: fit(tt.drop(index=t), MAIN)(tt.loc[t, 'sil'], tt.loc[t, 'k'])
                                  - tt.loc[t, GRID].to_numpy() for t in tt.index}, index=GRID).T
    k = sim.k.reindex(base.index)
    drop6 = base[0.0] - base[0.6]
    print('\n척도   평균 절대 오차(%p)  비율 0   비율 0.6   5클래스 과제의 부호 있는 오차   원본 정확도와 상관   0.6 하락과 상관   팀 분할 값')
    for kind in KINDS:
        e = err[kind]
        print(f'  {kind:<9} {e.abs().mean(axis=1).mean():6.2f}        {e[0.0].abs().mean():5.2f}   {e[0.6].abs().mean():5.2f}'
              f'        {e[k == 5].mean(axis=1).mean():+6.2f} ({int((k == 5).sum())}개)'
              f'              {spearmanr(val[kind], base[0.0])[0]:+.2f}             {spearmanr(val[kind], drop6)[0]:+.2f}'
              f'        {task_feature(feats[idx], y, kind):.3f}')
    a, b = err['mean'].abs().mean(axis=1), err['minclass'].abs().mean(axis=1)
    p = wilcoxon(a - b).pvalue
    ok = b.mean() <= 0.9 * a.mean() and p < 0.05 and (a - b).mean() > 0
    print(f'\n과제별 평균 절대 오차: mean {a.mean():.2f} 대 minclass {b.mean():.2f} '
          f'({(b.mean() / a.mean() - 1):+.0%}), Wilcoxon p={p:.3f}')
    worst = (a - b).sort_values()
    print('minclass가 가장 나빠진 과제: ' + ', '.join(f'{t} {v:+.2f}' for t, v in worst.head(3).items()))
    print('minclass가 가장 나아진 과제: ' + ', '.join(f'{t} {v:+.2f}' for t, v in worst.tail(3).items()))
    print(f'\n[판정] 3판 1단계: {"통과 -> 2단계(팀 분할 seed 6~10) 진행" if ok else "통과 못 함 -> 3판은 여기서 멈춤"}')
    if ok:
        flag.write_text('pass', encoding='utf-8')


if __name__ == '__main__':
    main()
