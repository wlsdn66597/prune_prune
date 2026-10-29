"""약한 클래스를 지키는 채널 선택: 학습 전에 무너질 클래스를 알고, 자를 때 그 클래스에 필요한 채널을 남긴다.

근거: 사전학습 분리도가 낮은 클래스가 프루닝 뒤 더 떨어진다(class_signals.py: 네 묶음 재현, 학습 후 신호보다 나음).
분리도 가중 회복(team_weighted.py)은 효과가 없었다. 용량이 모자란 상태에서 손실 가중치는 남은 채널의 쓰임만 바꿔
클래스 사이로 오차를 옮길 뿐이다. 그래서 '어떤 채널이 남을지'를 바꾼다. 비율 0.4~0.6에서는 movement와 L2 마스크가
거의 겹치지 않아도 전체 정확도가 같았으므로(마스크 자유도), 그 여유를 약한 클래스를 지키는 데 쓴다.
방법 (값은 하나로 고정, 고르지 않음):
  1. 파인튜닝한 원본에서 클래스마다 그 클래스 학습 이미지(증강, 최대 4배치)로 1차 Taylor 채널 점수를 잰다
     T_c = |Σ g·w| (출력 채널은 그 필터, 입력 채널은 그 입력 조각에 대해 합). 학습은 하지 않는다.
  2. conv마다 채널 방향으로 표준화한 T_c를 위험 가중치 w_c(team_weighted.py와 같은 분리도 가중치)로 더해 보호 점수를 만든다.
  3. 남길 채널 점수 = z(movement) + z(보호 점수). 자르기·BN 재보정·2800 step 회복은 표준과 같다.
비교와 판정은 team_weighted.py와 같다: 같은 원본(seed 0~5)의 표준 movement 모델과 비교하고, 비율마다 시드 평균으로
가장 약한 클래스(실루엣 최저) 재현율이 3%p 이상 오르고 전체 정확도가 0.5%p 넘게 떨어지지 않으면 충족.
0.4와 0.6 둘 다면 '효과 있음', 하나면 '부분'. 마스크가 movement와 얼마나 달라졌는지(출력 채널 기준 근사)도 낸다.

사용: python team_protect.py train && python team_protect.py judge > work/team_protect_result.txt
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import common as C
import prune_common as P
import team_weighted as TW
from plan_eval import REC_STEPS

N_BATCH = 4


def zs(x):
    return (x - x.mean()) / x.std().clamp_min(1e-12)


def class_taylor(model, train_ds, labels, device, workers, seed):
    """클래스마다 conv 채널의 1차 Taylor 점수 {클래스: ({conv: 출력 점수}, {conv: 입력 점수})}."""
    convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    out = {}
    model.eval()
    for c in np.unique(labels):
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(Subset(train_ds, np.flatnonzero(labels == c).tolist()), C.BATCH, shuffle=True,
                            num_workers=workers, generator=g)
        model.zero_grad()
        for k, (x, y) in enumerate(loader):
            if k >= N_BATCH:
                break
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
        o, i = {}, {}
        for n, m in convs.items():
            gw = (m.weight.grad * m.weight).detach().cpu()
            o[n], i[n] = gw.sum(dim=(1, 2, 3)).abs(), gw.sum(dim=(0, 2, 3)).abs()
        out[int(c)] = (o, i)
        del loader
    model.zero_grad(set_to_none=True)
    return out


def protect_scores(move, taylor, w):
    """movement 점수와 위험 가중 클래스별 Taylor 점수를 conv마다 표준화해 더한다."""
    b = move['best_i']
    mo = {n: v[b] for n, v in move['cum_out'].items()}
    mi = {n: v[b] for n, v in move['cum_in'].items()}
    out, inp = {}, {}
    for n in mo:
        po = sum(float(w[c]) * zs(taylor[c][0][n]) for c in taylor)
        pi = sum(float(w[c]) * zs(taylor[c][1][n]) for c in taylor)
        out[n], inp[n] = zs(mo[n]) + zs(po), zs(mi[n]) + zs(pi)
    return mo, out, inp


def keep_overlap(mo, new, r):
    """conv마다 출력 채널을 점수 상위 (1-r)로 남긴다고 할 때 movement와 새 점수가 같이 남기는 비율 (근사, 평균)."""
    ov = []
    for n in mo:
        k = max(1, int(len(mo[n]) * (1 - r)))
        a, b = set(torch.topk(mo[n], k).indices.tolist()), set(torch.topk(new[n], k).indices.tolist())
        ov.append(len(a & b) / k)
    return float(np.mean(ov))


def train(args, work):
    sil, w = TW.class_weights(args.work_dir, args.data_root)
    print('위험 가중치: ' + ', '.join(f'{c} {v:.2f}' for c, v in zip(C.CLASSES6, w)))
    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
    train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for s in args.seeds:
        base_dir = runs / 'team' / f'seed{s}'
        base, move = P.load_base(base_dir, len(C.CLASSES6), args.device), P.load_movement(base_dir)
        todo = [r for r in args.ratios if not (work / args.out_name / f'seed{s}' / f'mov_r{r}.json').exists()]
        if not todo:
            continue
        taylor = class_taylor(base, train_ds, np.asarray(tr_y), args.device, args.workers, s)
        mo, so, si = protect_scores(move, taylor, w)
        for r in todo:
            out = work / args.out_name / f'seed{s}' / f'mov_r{r}.json'
            t0 = time.time()
            C.set_seed(s)
            model = copy.deepcopy(base)
            P.prune(model, P.ChannelScoreImportance(model, so, si), r)
            P.recalibrate_bn(model, cache, tr_idx)
            P.recover(model, train_ds, args.steps, args.device, args.workers, s)
            info = {'seed': s, 'ratio': r, 'steps': args.steps, 'risk_weight': w.tolist(),
                    'keep_overlap_with_movement': keep_overlap(mo, so, r),
                    'val': C.metrics(C.predict_logits(model, cache, va_idx), va_y),
                    'test': C.metrics(C.predict_logits(model, cache, te_idx), te_y), 'time_sec': round(time.time() - t0, 1)}
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu(), out.with_suffix('.pt'))
            out.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
            print(f"seed{s} 비율 {r}: test {100 * info['test']['acc']:.2f}%, movement와 남기는 채널 겹침 "
                  f"{info['keep_overlap_with_movement']:.2f} ({info['time_sec']}s)", flush=True)
            del model


def judge(args, work):
    ov = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                       for p in (work / args.out_name).glob('seed*/mov_r*.json')])
    TW.judge(args, work, title='약한 클래스를 지키는 채널 선택', arm_name='보호')
    if not ov.empty:
        print('\n참고: movement 마스크와 남기는 출력 채널 겹침 (conv별 근사 평균) '
              + ', '.join(f'비율 {r:g} {v:.2f}' for r, v in ov.groupby('ratio').keep_overlap_with_movement.mean().items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['train', 'judge'])
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=TW.SEEDS)
    ap.add_argument('--ratios', type=float, nargs='+', default=TW.RATIOS)
    ap.add_argument('--steps', type=int, default=REC_STEPS)
    ap.add_argument('--out-name', default='team_protect', help='모델 폴더 (work/ 아래)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    work = C.runs_dir(args.work_dir).parent
    {'train': train, 'judge': judge}[args.step](args, work)


if __name__ == '__main__':
    main()
