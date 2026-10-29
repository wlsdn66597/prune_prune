"""무너질 클래스를 알려 주는 신호 비교와 나눠 본 재현 (학습 없음, 저장된 예측만 읽는다).

class_drop.py에서 '학습 전 사전학습 분리도가 낮은 클래스가 프루닝 뒤 더 떨어진다'가 나왔다(276조건, 과제 안 상관 −0.38).
이를 두 가지로 보강한다.
  [1] 나눠서 재현: 기존 과제 / 넓힌 과제(add_tasks.py) x 비율 0.6 / 0.8 네 묶음에서 따로 성립하나
  [2] 신호 비교: 같은 기준(과제 안 순위 상관, 가장 위험한 클래스 = 가장 많이 떨어진 클래스 일치율)으로 네 신호를 나란히 본다
        사전학습 분리도 (학습 전, 우리)         : 그 과제 학습 이미지에서 잰 클래스별 실루엣이 낮을수록 위험
        원본 검증 재현율 (학습 후)              : 낮을수록 위험 (원본에서 못 맞히던 클래스가 더 떨어진다는 가설)
        원본 결정 여유 (학습 후, Tran et al.식) : 검증 샘플의 '정답 logit − 가장 큰 오답 logit' 평균이 작을수록 위험
        클래스 크기 (Paganini식)               : 학습 장수가 적을수록 위험 (재질만인 과제에서는 모두 같아 정보 없음)
      학습 후 신호는 test가 아니라 원본의 검증 예측으로 계산한다 (하락은 test로 재므로 누수를 피한다).
      신호가 과제 안에서 모두 같으면 상관 0, 일치율은 우연과 같게 센다.
자료: class_drop.py와 같다 (prune_grid mov_cum 2800 step의 test 예측, 기준 모델 logits.npz).

판정 (결과를 보기 전에 정함, 상관은 3클래스 이상 과제만):
  재현 : 네 묶음 모두 사전학습 분리도의 과제 안 순위 상관(위험 방향)이 양수이고, 셋 이상에서 Wilcoxon p<0.05
  비교 : 사전학습 분리도의 평균 상관이 두 학습 후 신호(검증 재현율, 결정 여유) 각각보다 0.05 넘게 낮지 않으면
         '학습 전 신호로 충분', 둘 다보다 높고 짝 비교 p<0.05면 '학습 전 신호가 더 낫다'

사용: python class_signals.py > work/class_signals.txt
"""
import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import silhouette_samples

import common as C

RATIOS = [0.6, 0.8]
SIGNALS = {'presil': '사전학습 분리도 (학습 전)', 'valrec': '원본 검증 재현율', 'margin': '원본 결정 여유',
           'size': '클래스 크기'}
N_PERM = 1000


def per_class(y, logits, k):
    """클래스별 재현율(%)과 결정 여유(정답 logit − 가장 큰 오답 logit의 평균)."""
    logits = np.asarray(logits, dtype=float)
    rows = np.arange(len(y))
    other = logits.copy()
    other[rows, y] = -np.inf
    margin = logits[rows, y] - other.max(1)
    pred = logits.argmax(1)
    rec = np.array([100 * (pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(k)])
    mar = np.array([margin[y == c].mean() if (y == c).any() else np.nan for c in range(k)])
    return rec, mar


def load_rows(runs, images, tasks, feats):
    orig, rows, cache = set(C.build_tasks()), [], {}
    for r in RATIOS:
        for p in sorted(runs.glob(f'prune/*/fold*_seed0/mov_cum_r{r}_rec2800.npz')):
            t, f = p.parent.parent.name, int(re.match(r'fold(\d+)_', p.parent.name).group(1))
            bp = runs / 'base' / t / f'fold{f}_seed0' / 'logits.npz'
            if t not in tasks or not bp.exists():
                continue
            pr, bs = np.load(p), np.load(bp)
            if not np.array_equal(pr['test_idx'], bs['test_idx']):
                continue
            k = len(tasks[t]['classes'])
            if (t, f) not in cache:
                idx, ytr = C.task_split(images, tasks[t], f)['train']
                s = silhouette_samples(feats[idx], ytr, metric='cosine')
                vrec, vmar = per_class(bs['val_y'], bs['val_logits'], k)
                cache[(t, f)] = (np.array([s[ytr == c].mean() for c in range(k)]), vrec, vmar,
                                 np.bincount(ytr, minlength=k))
            sil, vrec, vmar, size = cache[(t, f)]
            drop = per_class(bs['test_y'], bs['test_logits'], k)[0] - per_class(pr['test_y'], pr['test_logits'], k)[0]
            for c in range(k):
                rows.append({'task': t, 'fold': f, 'ratio': r, 'subset': '기존' if t in orig else '넓힌', 'k': k,
                             'cls': tasks[t]['classes'][c], 'drop': drop[c],
                             'presil': -sil[c], 'valrec': -vrec[c], 'margin': -vmar[c], 'size': -float(size[c])})
    return pd.DataFrame(rows).dropna()


def group_arrays(d):
    return [(g[list(SIGNALS)].to_numpy(), g['drop'].to_numpy()) for _, g in d.groupby(['task', 'fold', 'ratio'])]


def hit(risk, drop):
    """가장 위험하다고 본 클래스(동점이면 나눠 셈)가 가장 많이 떨어진 클래스인 정도."""
    top_r = np.flatnonzero(np.isclose(risk, risk.max()))
    top_d = np.flatnonzero(np.isclose(drop, drop.max()))
    return len(np.intersect1d(top_r, top_d)) / len(top_r)


def stats(groups, j, rng, perm=True):
    """신호 j의 과제 안 상관(3클래스 이상, 하락이 모두 같은 과제 제외)과 일치율, 우연, 순열 p (perm=False면 생략)."""
    rhos = []
    for risk, drop in groups:
        if len(drop) >= 3 and np.ptp(drop) > 0:
            rhos.append(0.0 if np.ptp(risk[:, j]) == 0 else spearmanr(risk[:, j], drop)[0])
    obs = np.mean([hit(risk[:, j], drop) for risk, drop in groups])
    chance = np.mean([np.isclose(drop, drop.max()).sum() / len(drop) for _, drop in groups])
    if not perm:
        return np.array(rhos), obs, chance, np.nan
    null = [np.mean([hit(rng.permutation(risk[:, j]), drop) for risk, drop in groups]) for _ in range(N_PERM)]
    return np.array(rhos), obs, chance, (1 + sum(x >= obs for x in null)) / (1 + N_PERM)


def p_of(x):
    x = np.asarray(x)
    return wilcoxon(x).pvalue if len(x) > 5 and np.any(x != 0) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    runs = C.runs_dir(args.work_dir)
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    feats = np.load(C.meta_dir(args.work_dir) / 'feat_pretrained.npy')
    d = load_rows(runs, images, tasks, feats)
    rng = np.random.default_rng(0)
    print(f'=== 무너질 클래스를 알려 주는 신호 (조건 {d.groupby(["task", "fold", "ratio"]).ngroups}개, 클래스 행 {len(d)}개) ===')
    print('상관과 일치율은 모두 위험 방향 (양수·높을수록 그 신호가 무너질 클래스를 잘 가리킴)')

    print('\n[1] 나눠서 재현: 사전학습 분리도')
    rep = []
    for sub in ['기존', '넓힌']:
        for r in RATIOS:
            g = group_arrays(d[(d.subset == sub) & (d.ratio == r)])
            if not g:
                print(f'  {sub} 과제, 비율 {r:g}: 결과 없음')
                continue
            rhos, obs, chance, p_hit = stats(g, 0, rng)
            p = p_of(rhos)
            rep.append((rhos.mean() > 0, p < 0.05))
            print(f'  {sub} 과제, 비율 {r:g}: 조건 {len(g)}개  상관 {rhos.mean():+.2f} (n={len(rhos)}, p={p:.3f})  '
                  f'일치율 {obs:.1%} (우연 {chance:.1%}, 순열 p={p_hit:.3f})')

    print('\n[2] 신호 비교 (전체)')
    groups = group_arrays(d)
    res = {key: stats(groups, j, rng) for j, key in enumerate(SIGNALS)}
    base_rho = res['presil'][0]
    for key, (rhos, obs, chance, p_hit) in res.items():
        line = (f'  {SIGNALS[key]:<16} 상관 {rhos.mean():+.2f} (p={p_of(rhos):.3f})  '
                f'일치율 {obs:.1%} (우연 {chance:.1%}, 순열 p={p_hit:.3f})')
        if key != 'presil':
            line += f'  사전학습 분리도 − 이것: {base_rho.mean() - rhos.mean():+.2f} (짝 비교 p={p_of(base_rho - rhos):.3f})'
        print(line)
    for sub in ['기존', '넓힌']:
        g = group_arrays(d[d.subset == sub])
        if g:
            parts = [f'{SIGNALS[key].split(" (")[0]} {stats(g, j, rng, perm=False)[0].mean():+.2f}'
                     for j, key in enumerate(SIGNALS)]
            print(f'  {sub} 과제만 (상관): ' + ', '.join(parts))

    replicated = len(rep) == 4 and all(pos for pos, _ in rep) and sum(sig for _, sig in rep) >= 3
    post = ['valrec', 'margin']
    enough = all(base_rho.mean() >= res[k][0].mean() - 0.05 for k in post)
    better = all(base_rho.mean() > res[k][0].mean() and p_of(base_rho - res[k][0]) < 0.05 for k in post)
    print('\n[판정]')
    print(f'  재현 (네 묶음): {"예" if replicated else "아니오"}')
    print(f'  비교: {"학습 전 신호가 더 낫다" if better else ("학습 전 신호로 충분" if enough else "학습 후 신호가 더 낫다")}')


if __name__ == '__main__':
    main()
