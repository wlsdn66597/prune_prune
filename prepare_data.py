"""0단계-A: 이미지 목록, 평가용 캐시, 사전학습 특징, 중복 후보 리포트를 만든다.

사용: python prepare_data.py --data-root <클래스 폴더 6개가 들어 있는 dataset-resized 경로>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models

import common as C

BANDS = [(0.97, 1.01), (0.95, 0.97), (0.93, 0.95), (0.90, 0.93), (0.85, 0.90)]


def build_eval_cache(paths, workers):
    loader = DataLoader(C.EvalPreList(paths), batch_size=64, num_workers=workers)
    return torch.cat(list(loader))


@torch.no_grad()
def pretrained_features(x_uint8, device):
    """ImageNet 사전학습 ResNet-18(V1)의 마지막 특징 512차원, L2 정규화."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval().to(device)
    mean = torch.tensor(C.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(C.STD, device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(x_uint8), 256):
        xb = x_uint8[i:i + 256].to(device).float().div_(255)
        feats.append(m((xb - mean) / std).cpu())
    return F.normalize(torch.cat(feats), dim=1).numpy()


def nearest_pairs(images, feats):
    """클래스 안에서 각 이미지와 가장 비슷한 이미지 쌍(중복 제거)과 코사인 유사도."""
    rows = []
    for cls, sub in images.groupby('cls'):
        idx = sub['idx'].to_numpy()
        s = feats[idx] @ feats[idx].T
        np.fill_diagonal(s, -1)
        for a, b in enumerate(s.argmax(1)):
            i, j = sorted((int(idx[a]), int(idx[b])))
            rows.append((i, j, cls, float(s[a, b])))
    df = pd.DataFrame(rows, columns=['i', 'j', 'cls', 'sim']).drop_duplicates(['i', 'j'])
    rel = images['relpath'].to_numpy()
    df['relpath_i'], df['relpath_j'] = rel[df['i'].to_numpy()], rel[df['j'].to_numpy()]
    return df.sort_values('sim', ascending=False).reset_index(drop=True)


def save_sheets(pairs, x_uint8, out_dir, per_band=8, seed=0):
    """유사도 구간마다 쌍을 몇 개씩 뽑아 나란히 그린다. 같은 물체로 보이기 시작하는 구간을 찾는 용도."""
    rng = np.random.default_rng(seed)
    for lo, hi in BANDS:
        band = pairs[(pairs['sim'] >= lo) & (pairs['sim'] < hi)]
        if band.empty:
            continue
        pick = band.iloc[np.sort(rng.choice(len(band), min(per_band, len(band)), replace=False))]
        rows = (len(pick) + 1) // 2
        fig, axes = plt.subplots(rows, 4, figsize=(10, 2.8 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for k, (_, r) in enumerate(pick.iterrows()):
            for c, (i, name) in enumerate([(r['i'], r['relpath_i']), (r['j'], r['relpath_j'])]):
                ax = axes[k // 2, (k % 2) * 2 + c]
                ax.imshow(x_uint8[int(i)].permute(1, 2, 0).numpy())
                ax.set_title(Path(name).name + (f"\nsim={r['sim']:.3f}" if c == 0 else ''), fontsize=8)
        top = min(hi, 1.0)
        fig.suptitle(f'cosine {lo:.2f}-{top:.2f}: {len(pick)} of {len(band)} pairs')
        fig.tight_layout()
        fig.savefig(out_dir / f'dup_sheet_{lo:.2f}-{top:.2f}.png', dpi=80)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True, help='클래스 폴더 6개가 들어 있는 dataset-resized 경로')
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    md = C.meta_dir(args.work_dir)
    md.mkdir(parents=True, exist_ok=True)

    # 1) 이미지 목록: ImageFolder 순서를 그대로 써서 노트북과 인덱스를 맞춘다
    ds = datasets.ImageFolder(str(root))
    if ds.classes != C.CLASSES6:
        raise SystemExit(f'클래스 폴더가 예상과 다릅니다: {ds.classes}')
    images = pd.DataFrame({
        'idx': np.arange(len(ds.samples)),
        'relpath': [Path(p).relative_to(root).as_posix() for p, _ in ds.samples],
        'cls': [ds.classes[t] for _, t in ds.samples],
        'label6': [t for _, t in ds.samples]})
    images.to_csv(md / 'images.csv', index=False)
    (md / 'config.json').write_text(json.dumps({'data_root': str(root)}, ensure_ascii=False), encoding='utf-8')
    print('클래스별 장수:', images['cls'].value_counts().reindex(C.CLASSES6).to_dict())

    # 2) 평가용 캐시 (Resize 256 -> CenterCrop 224, uint8)
    x = build_eval_cache([str(root / r) for r in images['relpath']], args.workers)
    torch.save(x, md / 'eval224_uint8.pt')
    print(f'평가 캐시: {tuple(x.shape)}, {x.numel() / 1e6:.0f} MB')

    # 3) 사전학습 특징 (중복 확인과 과제 유사도 계산에 쓴다)
    feats = pretrained_features(x, args.device)
    np.save(md / 'feat_pretrained.npy', feats)

    # 4) 중복 후보: 클래스 안 최근접 이웃 쌍
    pairs = nearest_pairs(images, feats)
    pairs.to_csv(md / 'dup_nn.csv', index=False)
    q = pairs['sim'].quantile([0.5, 0.9, 0.99]).round(3).to_dict()
    print('최근접 이웃 유사도 분위수(50/90/99%):', q, ' 최대:', round(float(pairs['sim'].max()), 3))
    for t in (0.90, 0.93, 0.95, 0.97):
        print(f'  유사도 >= {t:.2f}: {int((pairs["sim"] >= t).sum())}쌍')
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(pairs['sim'], bins=60)
    ax.set_xlabel('nearest-neighbor cosine (same class)')
    fig.tight_layout()
    fig.savefig(md / 'dup_hist.png', dpi=100)
    plt.close(fig)
    save_sheets(pairs, x, md)
    print(f'{md} 의 dup_sheet_*.png를 열어 같은 물체로 보이기 시작하는 유사도 구간을 확인하세요.')


if __name__ == '__main__':
    main()
