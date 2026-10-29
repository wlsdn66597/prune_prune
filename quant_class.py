"""양자화도 같은 클래스를 먼저 깨나 (학습 없음, 저장된 모델을 양자화해 평가만 한다).

프루닝에서는 학습 전 사전학습 분리도가 낮은 클래스일수록 더 무너졌다(class_signals.py, 과제 안 상관 +0.38).
팀 파이프라인은 프루닝 뒤 양자화를 하므로, 양자화가 만드는 추가 손실도 같은 클래스에 몰리는지 본다.
같으면 '압축은 프루닝이든 양자화든 덜 갈라진 클래스부터 깬다'를 팀 전체 메시지로 쓸 수 있다.
양자화: 예산 분할 실험(split_eval.py)과 같다. conv 가중치만, 출력 채널별 대칭, 튀는 값 잘라내기(mse),
  양자화 뒤 BN 재보정. 비트는 W4와 mix5(4비트 오차가 큰 필터부터 8비트로 올려 평균 5비트).
손실은 같은 모델의 fp32와 양자화 뒤의 클래스별 test 재현율 차이다 (프루닝 손실은 빼고 양자화 손실만 본다).
자료: 22과제 x fold 0·1의 원본(비율 0)과 movement로 자른 모델(work/runs/qready, 비율 0.6·0.8),
      팀 분할 6분류 원본과 비율 0.4 모델(seed 0~5).

판정 (결과를 보기 전에 정함): 자른 모델(0.6·0.8)의 W4에서, 과제 안 순위 상관(분리도가 낮을수록 손실이 큰 방향)의
  평균이 양수이고 Wilcoxon p<0.05이며, 분리도 최저 클래스 = 손실 최대 클래스 일치율이 우연보다 높으면(순열 p<0.05)
  '양자화도 같은 클래스를 먼저 깬다'. 원본(비율 0)과 mix5는 참고로 낸다.

사용: python quant_class.py > work/quant_class.txt   (GPU, 약 10~15분)
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import silhouette_samples

import common as C
import prune_common as P
from qready_eval import plans_for, quantized

BITS = ['W4', 'mix5']
N_PERM = 1000


def recalls(model, cache, idx, y, k):
    pred = C.predict_logits(model, cache, idx).argmax(1).numpy()
    return np.array([100 * (pred[y == c] == c).mean() for c in range(k)])


def quant_drops(model, cache, split, k):
    """{비트: 클래스별 재현율 손실(fp32 - 양자화)}."""
    te_idx, te_y = split['test']
    base = recalls(model, cache, te_idx, te_y, k)
    plans, out = plans_for(model, 'mse'), {}
    for b in BITS:
        qm = quantized(model, plans[b], 'mse')
        P.recalibrate_bn(qm, cache, split['train'][0])
        out[b] = base - recalls(qm, cache, te_idx, te_y, k)
        del qm
    return out


def class_sil(feats, idx, y, k):
    s = silhouette_samples(feats[idx], y, metric='cosine')
    return np.array([s[y == c].mean() for c in range(k)])


def judge_groups(d, rng):
    """과제 안 상관(3클래스 이상)과 일치율·우연·순열 p. 위험 = -분리도."""
    groups = [(-g['sil'].to_numpy(), g['drop'].to_numpy()) for _, g in d.groupby(['task', 'fold'])]

    def hit(risk, drop):
        top_r = np.flatnonzero(np.isclose(risk, risk.max()))
        return len(np.intersect1d(top_r, np.flatnonzero(np.isclose(drop, drop.max())))) / len(top_r)

    rhos = [spearmanr(r, dr)[0] for r, dr in groups if len(dr) >= 3 and np.ptp(dr) > 0]
    obs = np.mean([hit(r, dr) for r, dr in groups])
    chance = np.mean([np.isclose(dr, dr.max()).sum() / len(dr) for _, dr in groups])
    null = [np.mean([hit(rng.permutation(r), dr) for r, dr in groups]) for _ in range(N_PERM)]
    p_rho = wilcoxon(rhos).pvalue if len(rhos) > 5 else np.nan
    return np.mean(rhos), p_rho, len(rhos), obs, chance, (1 + sum(x >= obs for x in null)) / (1 + N_PERM)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.0, 0.6, 0.8])
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    runs = C.runs_dir(args.work_dir)
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    feats = np.load(C.meta_dir(args.work_dir) / 'feat_pretrained.npy')
    cache = C.EvalCache(args.work_dir, args.device)

    rows = []
    for t in C.build_tasks():
        k = len(tasks[t]['classes'])
        for f in args.folds:
            split = C.task_split(images, tasks[t], f)
            sil = class_sil(feats, *split['train'], k)
            for r in args.ratios:
                if r == 0:
                    model = P.load_base(runs / 'base' / t / f'fold{f}_seed0', k, args.device)
                else:
                    p = runs / 'qready' / t / f'fold{f}_seed0' / f'movement_r{r}.pt'
                    if not p.exists():
                        continue
                    model = torch.load(p, map_location=args.device, weights_only=False).eval()
                for b, dr in quant_drops(model, cache, split, k).items():
                    for c in range(k):
                        rows.append({'task': t, 'fold': f, 'ratio': r, 'bits': b, 'cls': tasks[t]['classes'][c],
                                     'sil': sil[c], 'drop': dr[c]})
                del model
        print(f'{t} 완료', flush=True)
    d = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    print(f'\n=== 양자화가 먼저 깨는 클래스 (양자화 손실만, 과제 {d.task.nunique()}개 x fold {args.folds}) ===')
    print('상관과 일치율은 위험 방향 (양수·높을수록 분리도가 낮은 클래스가 더 많이 깨짐)')
    verdict = {}
    for b in BITS:
        for r in args.ratios:
            s = d[(d.bits == b) & (d.ratio == r)]
            if s.empty:
                continue
            rho, p, n, obs, chance, ph = judge_groups(s, rng)
            print(f'  {b:<5} 비율 {r:g}: 평균 손실 {s["drop"].mean():5.2f}%p  상관 {rho:+.2f} (n={n}, p={p:.3f})  '
                  f'일치율 {obs:.1%} (우연 {chance:.1%}, 순열 p={ph:.3f})')
        s = d[(d.bits == b) & (d.ratio > 0)]
        if not s.empty:
            verdict[b] = judge_groups(s, rng)
            rho, p, n, obs, chance, ph = verdict[b]
            print(f'  {b:<5} 자른 모델 합계: 상관 {rho:+.2f} (n={n}, p={p:.3f})  일치율 {obs:.1%} (우연 {chance:.1%}, 순열 p={ph:.3f})')
    print('\n재질별 양자화 손실 (자른 모델, W4, 과제 안 평균을 뺀 값 %p)')
    s = d[(d.bits == 'W4') & (d.ratio > 0)].copy()
    s['c'] = s['drop'] - s.groupby(['task', 'fold', 'ratio'])['drop'].transform('mean')
    print('  ' + ', '.join(f'{c} {v:+.2f}' for c, v in s.groupby('cls').c.mean().sort_values(ascending=False).items()))

    split = C.team_split(images)
    sil = class_sil(feats, *split['train'], len(C.CLASSES6))
    team = []
    for s_ in range(6):
        folder = 'team_plan' if s_ < 3 else 'team_plan2'
        models = {0.0: P.load_base(runs / 'team' / f'seed{s_}', len(C.CLASSES6), args.device)}
        p = runs.parent / folder / f'seed{s_}' / 'mov_r0.4.pt'
        if p.exists():
            models[0.4] = torch.load(p, map_location=args.device, weights_only=False).eval()
        for r, m in models.items():
            for b, dr in quant_drops(m, cache, split, len(C.CLASSES6)).items():
                team += [{'seed': s_, 'ratio': r, 'bits': b, 'cls': c, 'drop': dr[i]} for i, c in enumerate(C.CLASSES6)]
    tt = pd.DataFrame(team).pivot_table(index='cls', columns=['ratio', 'bits'], values='drop').reindex(C.CLASSES6)
    tt.columns = [f'비율 {r:g} {b}' for r, b in tt.columns]
    tt.insert(0, '분리도', sil)
    print('\n팀 분할 6분류: 양자화 손실 (%p, 시드 0~5 평균, 행은 클래스)')
    print(tt.sort_values('분리도').round(2).to_string())

    rho, p, _, obs, chance, ph = verdict.get('W4', (np.nan,) * 6)
    ok = rho > 0 and p < 0.05 and obs > chance and ph < 0.05
    print('\n[판정] 양자화도 같은 클래스를 먼저 깬다 (자른 모델, W4): ' + ('예' if ok else '아니오'))


if __name__ == '__main__':
    main()
