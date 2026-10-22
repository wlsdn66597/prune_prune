"""프루닝 공통 코드: 중요도 기준, one-shot 프루닝, BN 재보정, 회복 학습.

Torch-Pruning은 버전마다 클래스 이름이 조금씩 달라서 여러 이름을 차례로 찾는다.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torch.utils.data import DataLoader

import common as C

CRITERIA = ['random', 'l2', 'bn', 'taylor', 'mov_ckpt', 'mov_cum']
EXTRA_CRITERIA = ['mov_late']  # 첫 평가 구간(헤드가 무작위인 초반)을 뺀 누적 movement


def _find(module, *names):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    raise AttributeError(f'torch_pruning에 {names} 중 어느 것도 없습니다. 버전을 확인하세요.')


def _idx_list(idxs):
    return [int(getattr(i, 'idx', i)) for i in idxs]


class ChannelScoreImportance(tp.importance.Importance):
    """conv 이름별로 미리 계산한 채널 점수(출력 방향, 입력 방향)를 그룹 단위로 평균한다.

    movement 기준에 쓴다. 점수는 자르기 전 모델 기준이라 one-shot 프루닝에만 맞다.
    """

    def __init__(self, model, out_scores, in_scores):
        self.name_of = {m: n for n, m in model.named_modules()}
        self.out, self.inp = out_scores, in_scores

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        scores = []
        for dep, idxs in group:
            layer = dep.target.module
            name = self.name_of.get(layer)
            if not isinstance(layer, nn.Conv2d) or name not in self.out:
                continue
            fn = getattr(dep.handler, '__name__', '')
            idx = torch.as_tensor(_idx_list(idxs), dtype=torch.long)
            if fn == 'prune_out_channels':
                scores.append(self.out[name][idx])
            elif fn == 'prune_in_channels':
                scores.append(self.inp[name][idx])
        if not scores:
            return None
        n0 = len(scores[0])
        return torch.stack([s for s in scores if len(s) == n0]).mean(0).float()


def load_base(run_dir, num_classes, device):
    m = C.build_model(num_classes, pretrained=False)
    m.load_state_dict(torch.load(Path(run_dir) / 'model.pt', map_location='cpu'))
    return m.to(device).eval()


def load_movement(run_dir):
    return torch.load(Path(run_dir) / 'movement.pt', map_location='cpu', weights_only=False)


def make_importance(crit, model, move):
    if crit == 'random':
        return _find(tp.importance, 'RandomImportance')()
    if crit == 'l2':
        return _find(tp.importance, 'GroupMagnitudeImportance', 'MagnitudeImportance')(p=2)
    if crit == 'bn':
        return _find(tp.importance, 'BNScaleImportance', 'GroupBNScaleImportance')()
    if crit == 'taylor':
        return _find(tp.importance, 'GroupTaylorImportance', 'TaylorImportance')()
    b = move['best_i']
    if crit == 'mov_ckpt':
        o, i = move['ckpt_out'], move['ckpt_in']
    elif crit == 'mov_cum':
        o = {n: v[b] for n, v in move['cum_out'].items()}
        i = {n: v[b] for n, v in move['cum_in'].items()}
    elif crit == 'mov_late':
        o = {n: v[b] - v[0] for n, v in move['cum_out'].items()}
        i = {n: v[b] - v[0] for n, v in move['cum_in'].items()}
    else:
        raise ValueError(f'모르는 기준: {crit}')
    return ChannelScoreImportance(model, o, i)


def accumulate_grads(model, train_ds, device, n_batches, seed, workers):
    """Taylor 중요도용 기울기. 증강을 켠 학습 배치로 계산한다 (BN 통계를 건드리지 않게 eval 모드)."""
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, C.BATCH, shuffle=True, num_workers=workers, generator=g)
    model.eval()
    model.zero_grad()
    for k, (x, y) in enumerate(loader):
        if k >= n_batches:
            break
        F.cross_entropy(model(x.to(device)), y.to(device)).backward()
    del loader


def prune(model, importance, ratio, ignore_conv1=False, ratio_dict=None):
    """모든 그룹을 같은 비율로 자른다 (global pruning 아님). fc 출력은 항상 보호한다.

    ratio_dict({모듈: 비율})를 주면 그 모듈이 시작점인 그룹만 그 비율로 자른다 (6단계에서 ratio=0과 함께 쓴다).
    """
    ignored = [model.fc] + ([model.conv1] if ignore_conv1 else [])
    ex = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    Pruner = _find(tp.pruner, 'MetaPruner', 'BasePruner')
    kw = {'importance': importance, 'ignored_layers': ignored}
    try:
        pruner = Pruner(model, ex, pruning_ratio=ratio, pruning_ratio_dict=ratio_dict, **kw)
    except TypeError:  # 옛 버전은 ch_sparsity
        pruner = Pruner(model, ex, ch_sparsity=ratio, ch_sparsity_dict=ratio_dict, **kw)
    pruner.step()
    model.zero_grad(set_to_none=True)
    return model


def prune_global(model, importance, ratio, max_ratio=0.95):
    """Torch-Pruning 전역 프루닝: 정규화한 중요도로 모든 그룹을 함께 순위 매겨 자른다 (그룹당 최대 max_ratio)."""
    ex = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    Pruner = _find(tp.pruner, 'MetaPruner', 'BasePruner')
    kw = {'importance': importance, 'ignored_layers': [model.fc], 'global_pruning': True}
    options = [{'pruning_ratio': ratio, 'max_pruning_ratio': max_ratio},
               {'ch_sparsity': ratio, 'max_ch_sparsity': max_ratio},
               {'pruning_ratio': ratio}]
    for extra in options:
        try:
            pruner = Pruner(model, ex, **kw, **extra)
            break
        except TypeError:
            continue
    else:
        raise RuntimeError('전역 프루닝 설정을 만들 수 없습니다. torch-pruning 버전을 확인하세요.')
    pruner.step()
    model.zero_grad(set_to_none=True)
    return model


@torch.no_grad()
def recalibrate_bn(model, cache, idx):
    """BN 통계만 다시 잡는다 (EagleEye, ECCV 2020). 증강 없는 학습 이미지를 train 모드로 통과시킨다."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # 누적 평균
    model.train()
    for xb in cache.batches(idx, bs=64):
        model(xb)
    for m, v in zip(bns, momenta):
        m.momentum = v
    model.eval()


def recover(model, train_ds, steps, device, workers, seed):
    """회복 학습: 노트북과 같은 AdamW, 학습률을 cosine으로 1e-4에서 0까지 줄이고 마지막 모델을 쓴다."""
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, C.BATCH, shuffle=True, num_workers=workers, generator=g,
                        pin_memory=device != 'cpu', persistent_workers=workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    crit = nn.CrossEntropyLoss()
    step = 0
    while step < steps:
        for x, y in loader:
            model.train()
            opt.zero_grad(set_to_none=True)
            crit(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True)).backward()
            opt.step()
            sched.step()
            step += 1
            if step >= steps:
                break
    del loader
    model.eval()


def model_stats(model):
    ex = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    macs, params = tp.utils.count_ops_and_params(model, ex)
    channels = {n: m.out_channels for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    return {'macs': int(macs), 'params': int(params), 'channels': channels}


def save_result(path, info, logits=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if logits is not None:
        np.savez_compressed(path.with_suffix('.npz'), **logits)
    path.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')  # 완료 표시로 마지막에 쓴다
