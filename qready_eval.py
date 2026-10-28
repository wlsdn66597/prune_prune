"""1단계 강건성 확인: 저장한 모델을 다시 학습하지 않고, 측정용 양자화를 더 좋은 방식으로 바꿔도 이점이 남는지 본다.

1단계의 측정용 양자화(최댓값 눈금)는 튀는 값에 가장 약한 방식이라, 우리 마스크에 유리했을 수 있다.
변형 (모두 conv 가중치만, 출력 채널별 대칭, fc·BN 그대로):
  max  : 눈금 = 필터 최댓값 (1단계와 같음)
  mse  : 필터마다 최댓값의 50~100% 중 양자화 오차가 가장 작은 지점에서 잘라 눈금을 정한다 (튀는 값을 잘라내는 흔한 개선)
  비트 : W4 (모두 4비트), mix5·mix6 (해당 방식의 4비트 오차가 큰 필터부터 8비트로 올려 평균 5·6비트, 같은 너비면 같은 크기)
  +recal : 양자화한 뒤 BN 통계를 학습 사진으로 다시 잡는다 (학습 없음)

판정 (결과를 보기 전에 정함, 비율 0.8):
  qr_full@1.0 - movement가 mse-mix5와 max-mix5+recal 둘 다에서 양수이고 Wilcoxon p<0.05이면 '양자화 방식이 바뀌어도 유지',
  아니면 '최댓값 눈금 양자화에서만 확인된 효과'

사용: python qready_eval.py > work/qready_eval.txt
"""
import argparse
import copy
import json

import pandas as pd
import torch
import torch.nn as nn

import common as C
import prune_common as P
from qready_train import ARMS
from summarize_budget import fmt, stat

CLIPS = torch.linspace(0.5, 1.0, 11).tolist()
CONFIGS = ['max-W4', 'max-mix5', 'max-mix6', 'max-mix5+recal', 'mse-W4', 'mse-mix5', 'mse-mix6', 'mse-mix5+recal']


def fake_quant(W, bits, mode):
    """출력 채널별 대칭 양자화. bits: 필터별 비트 [출력]. mode='max' 또는 'mse'(필터별 오차 최소 자르기)."""
    qmax = (2 ** (bits.to(W.device).float() - 1) - 1).view(-1, 1, 1, 1)
    amax = W.abs().amax(dim=(1, 2, 3), keepdim=True)

    def q(scale):
        s = scale.clamp_min(1e-12)
        return torch.minimum(torch.maximum((W / s).round(), -qmax), qmax) * s

    if mode == 'max':
        return q(amax / qmax)
    best, best_err = None, None
    for c in CLIPS:
        Wq = q(c * amax / qmax)
        err = ((W - Wq) ** 2).sum(dim=(1, 2, 3), keepdim=True)
        if best is None:
            best, best_err = Wq, err
        else:
            take = err < best_err
            best, best_err = torch.where(take, Wq, best), torch.where(take, err, best_err)
    return best


def plans_for(model, mode):
    """W4, mix5, mix6의 필터별 비트. 8비트로 올릴 순서는 해당 방식의 4비트 상대 오차가 큰 순."""
    convs = {n: m.weight.detach().float() for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    items = []
    for n, W in convs.items():
        Wq = fake_quant(W, torch.full((W.shape[0],), 4), mode)
        rel = ((W - Wq) ** 2).sum(dim=(1, 2, 3)) / (W ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
        items += [(r, n, j, W[0].numel()) for j, r in enumerate(rel.tolist())]
    total = sum(p for *_, p in items)
    plans = {'W4': {n: torch.full((W.shape[0],), 4) for n, W in convs.items()}}
    for avg in (5, 6):
        budget = (avg - 4) * total
        mix = {n: torch.full((W.shape[0],), 4) for n, W in convs.items()}
        for _, n, j, p in sorted(items, reverse=True):
            if 4 * p <= budget:
                mix[n][j] = 8
                budget -= 4 * p
        plans[f'mix{avg}'] = mix
    return plans


@torch.no_grad()
def quantized(model, plan, mode):
    q = copy.deepcopy(model)
    for n, m in q.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = fake_quant(m.weight.data, plan[n], mode)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--out-name', default='qready')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    cache = C.EvalCache(args.work_dir, args.device)
    root = C.runs_dir(args.work_dir) / args.out_name
    paths = sorted(root.glob('*/fold*_seed0/*.pt'))
    rows, splits = [], {}
    for k, path in enumerate(paths, 1):
        j = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        key = (j['task'], j['fold'])
        if key not in splits:
            splits[key] = C.task_split(images, tasks[j['task']], j['fold'])
        tr_idx = splits[key]['train'][0]
        te_idx, te_y = splits[key]['test']
        model = torch.load(path, map_location=args.device, weights_only=False).eval()
        acc = lambda m: 100 * C.metrics(C.predict_logits(m, cache, te_idx), te_y)['acc']
        row = {'task': j['task'], 'fold': j['fold'], 'ratio': j['ratio'], 'arm': j['arm'], 'fp32': acc(model)}
        for mode in ('max', 'mse'):
            for pname, plan in plans_for(model, mode).items():
                qm = quantized(model, plan, mode)
                row[f'{mode}-{pname}'] = acc(qm)
                if pname == 'mix5':
                    P.recalibrate_bn(qm, cache, tr_idx)
                    row[f'{mode}-mix5+recal'] = acc(qm)
        rows.append(row)
        if k % 20 == 0:
            print(f'[{k}/{len(paths)}]', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.runs_dir(args.work_dir) / 'qready_eval.csv', index=False)
    print(f'=== 1단계 강건성 확인: 저장 모델 {len(df)}개, test 정확도 % ===')
    verdict = {}
    for r in sorted(df.ratio.unique()):
        d = df[df.ratio == r]
        arms = [a for a in ARMS.get(r, []) if a in set(d.arm)]
        print(f'\n==================== 비율 {r} ====================')
        print(d.groupby('arm')[['fp32'] + CONFIGS].mean().loc[arms].round(2).T.to_string())
        w = d.pivot_table(index=['task', 'fold'], columns='arm', values=['fp32'] + CONFIGS)
        print('\n  movement 대비 차이 (%p, 같은 과제·fold): 평균 [승/무/패] Wilcoxon p')
        for arm in [a for a in arms if a != 'movement']:
            print(f'  {arm}')
            for c in ['fp32'] + CONFIGS:
                dd = (w[(c, arm)] - w[(c, 'movement')]).dropna()
                print(f'    {c:<15} {fmt(*stat(dd))}  n={len(dd)}')
                if r == 0.8 and arm == 'qr_full@1.0':
                    verdict[c] = stat(dd)
    if verdict:
        ok = all(verdict[c][0] > 0 and verdict[c][2] < 0.05 for c in ('mse-mix5', 'max-mix5+recal'))
        print('\n[판정] 비율 0.8, qr_full@1.0 - movement: '
              + ', '.join(f'{c} {verdict[c][0]:+.2f}%p (p={verdict[c][2]:.3f})' for c in ('mse-mix5', 'max-mix5+recal')))
        print('  => ' + ('양자화 방식이 바뀌어도 유지' if ok else '최댓값 눈금 양자화에서만 확인된 효과'))


if __name__ == '__main__':
    main()
