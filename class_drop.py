"""무너지는 클래스를 학습 전에 알 수 있나 (학습 없음, 저장된 test 예측만 읽는다).

팀 분할 6분류에서 metal(원본 재현율 95%)이 비율 0.4에서 −10.8%p로 가장 먼저 무너졌다. 이것이 '원래 헷갈리는 클래스가
무너진다'(학습 전 분리도로 알 수 있음)인지, '특정 재질이 프루닝에 약하다'(클래스 정체로 알 수 있음)인지 가른다.
과제마다 클래스별 test 재현율 하락(원본 - 자른 모델)을, 그 과제 학습 이미지에서 잰 클래스별 실루엣
(사전학습 ResNet-18 특징, 코사인, 그 클래스 샘플들의 평균)과 비교한다.
자료: prune_grid mov_cum 2800 step의 test 예측(npz)과 기준 모델 logits.npz, 비율 0.6·0.8.
  (기존 과제는 fold 2~4 + fold 0·1의 0.8, 넓힌 과제는 fold 0·1. all6(교차검증 6분류)도 포함)

판정 (결과를 보기 전에 정함):
  학습 전 경고 가능: 과제 안에서 실루엣이 가장 낮은 클래스가 가장 많이 떨어진 클래스와 일치하는 비율이
                     우연(평균 1/k)보다 높고(과제 안에서 실루엣을 섞는 순열 검정 p<0.05),
                     3클래스 이상 과제의 과제 안 순위 상관(실루엣 vs 하락) 평균이 음수(Wilcoxon p<0.05)
  재질 고유 약점   : 과제 안 평균을 뺀 하락(중심화)을 재질별로 모아, metal이 가장 크고 0보다 크면(Wilcoxon p<0.05)
                     'metal은 프루닝에 약하다'
test 장수가 적은 클래스(trash는 fold당 약 27장)는 하락이 들쭉날쭉하다.

사용: python class_drop.py > work/class_drop.txt
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import silhouette_samples

import common as C

RATIOS = [0.6, 0.8]
N_PERM = 2000


def recalls(y, logits, k):
    pred = np.asarray(logits).argmax(1)
    return np.array([100 * (pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(k)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    runs = C.runs_dir(args.work_dir)
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    feats = np.load(C.meta_dir(args.work_dir) / 'feat_pretrained.npy')

    sil_cache, rows = {}, []
    for r in RATIOS:
        for p in sorted(runs.glob(f'prune/*/fold*_seed0/mov_cum_r{r}_rec2800.npz')):
            t, f = p.parent.parent.name, int(re.match(r'fold(\d+)_', p.parent.name).group(1))
            bp = runs / 'base' / t / f'fold{f}_seed0' / 'logits.npz'
            if t not in tasks or not bp.exists():
                continue
            pr, bs = np.load(p), np.load(bp)
            if not np.array_equal(pr['test_idx'], bs['test_idx']):
                print(f'test 이미지가 달라 건너뜀: {t} fold{f}')
                continue
            classes, y = tasks[t]['classes'], pr['test_y']
            if (t, f) not in sil_cache:
                idx, ytr = C.task_split(images, tasks[t], f)['train']
                s = silhouette_samples(feats[idx], ytr, metric='cosine')
                sil_cache[(t, f)] = [s[ytr == c].mean() for c in range(len(classes))]
            b, a = recalls(y, bs['test_logits'], len(classes)), recalls(y, pr['test_logits'], len(classes))
            for c, name in enumerate(classes):
                rows.append({'task': t, 'fold': f, 'ratio': r, 'k': len(classes), 'cls': name, 'n': int((y == c).sum()),
                             'sil': sil_cache[(t, f)][c], 'base': b[c], 'drop': b[c] - a[c]})
    d = pd.DataFrame(rows).dropna()
    d['cdrop'] = d['drop'] - d.groupby(['task', 'fold', 'ratio'])['drop'].transform('mean')
    groups = list(d.groupby(['task', 'fold', 'ratio']))
    print(f'=== 무너지는 클래스를 학습 전에 알 수 있나 (조건 {len(groups)}개, 클래스 행 {len(d)}개) ===')

    def hit_rate(sil_of):
        return np.mean([g['drop'].to_numpy()[np.argmin(sil_of(g))] == g['drop'].max() for _, g in groups])

    obs = hit_rate(lambda g: g['sil'].to_numpy())
    chance = np.mean([1 / len(g) for _, g in groups])
    rng = np.random.default_rng(0)
    null = [hit_rate(lambda g: rng.permutation(g['sil'].to_numpy())) for _ in range(N_PERM)]
    p_hit = (1 + sum(x >= obs for x in null)) / (1 + N_PERM)
    rho = pd.Series({key: spearmanr(g['sil'], g['drop'])[0] for key, g in groups if len(g) >= 3}).dropna()
    p_rho = wilcoxon(rho).pvalue if len(rho) > 5 else np.nan
    print('\n[1] 과제 안에서 실루엣이 가장 낮은 클래스 = 가장 많이 떨어진 클래스?')
    print(f'  일치율 {obs:.1%} (우연 {chance:.1%}, 순열 검정 p={p_hit:.3f})')
    print(f'  3클래스 이상 과제의 과제 안 순위 상관(실루엣 vs 하락): 평균 {rho.mean():+.2f} '
          f'(n={len(rho)}, Wilcoxon p={p_rho:.3f}, 음수면 덜 갈라진 클래스가 더 떨어짐)')
    rho_b = pd.Series({key: spearmanr(g['base'], g['drop'])[0] for key, g in groups if len(g) >= 3}).dropna()
    print(f'  참고: 원본 재현율 vs 하락의 과제 안 순위 상관 평균 {rho_b.mean():+.2f} (음수면 잘 맞히던 클래스가 덜 떨어짐)')

    print('\n[2] 재질별 하락 (과제 안 평균을 뺀 값, %p, 양수면 그 과제의 다른 클래스보다 더 떨어짐)')
    mat = d.groupby('cls').agg(중심화하락=('cdrop', 'mean'), 하락=('drop', 'mean'), 실루엣=('sil', 'mean'),
                               원본재현율=('base', 'mean'), 행=('cdrop', 'size'))
    mat['p'] = [wilcoxon(d.loc[d.cls == c, 'cdrop']).pvalue if (d.cls == c).sum() > 5 else np.nan for c in mat.index]
    print(mat.sort_values('중심화하락', ascending=False).round(3).to_string())
    for r in RATIOS:
        m = d[d.ratio == r].groupby('cls').cdrop.mean().sort_values(ascending=False)
        print(f'  비율 {r:g}: ' + ', '.join(f'{c} {v:+.2f}' for c, v in m.items()))
    six = d[d.task == 'all6'].groupby(['ratio', 'cls'])['drop'].mean().unstack('ratio')
    if not six.empty:
        print('\n  참고: all6(교차검증 6분류)의 클래스별 하락 (%p)')
        print(six.round(1).to_string())

    idx, y = C.team_split(images)['train']
    s = silhouette_samples(feats[idx], y, metric='cosine')
    team = pd.Series({c: s[y == i].mean() for i, c in enumerate(C.CLASSES6)}).sort_values()
    print('\n[3] 팀 분할 6분류 학습 이미지의 클래스별 실루엣 (낮을수록 헷갈림): '
          + ', '.join(f'{c} {v:+.3f}' for c, v in team.items()))
    print('    (팀 분할 0.4에서 많이 떨어진 순서: metal −10.8, glass −6.1, paper −2.2, cardboard·plastic 0, trash +1.7)')

    warn = obs > chance and p_hit < 0.05 and rho.mean() < 0 and p_rho < 0.05
    weak = mat['중심화하락'].idxmax() == 'metal' and mat.loc['metal', '중심화하락'] > 0 and mat.loc['metal', 'p'] < 0.05
    print('\n[판정]')
    print(f'  학습 전 경고 가능 (덜 갈라진 클래스가 먼저 무너짐): {"예" if warn else "아니오"}')
    print(f'  재질 고유 약점 (metal이 프루닝에 가장 약함): {"예" if weak else "아니오"} '
          f'(가장 많이 떨어지는 재질: {mat["중심화하락"].idxmax()})')
    (runs.parent / 'class_drop.json').write_text(json.dumps({'warn': bool(warn), 'metal_weak': bool(weak)}),
                                                  encoding='utf-8')


if __name__ == '__main__':
    main()
