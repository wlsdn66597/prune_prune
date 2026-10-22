"""0단계-B: 중복 그룹 -> 이미지별 fold 고정, 재질 클래스 장수 맞추기, 과제 목록, 과제별 유사도.

사용: python make_folds.py --dup-thr 0.95
      (중복 그룹을 만들지 않으려면 --dup-thr 1.01)
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedGroupKFold

import common as C


def dup_groups(images, feats, thr):
    """같은 클래스 안에서 코사인 유사도가 thr 이상인 이미지를 같은 그룹으로 묶는다 (union-find)."""
    parent = np.arange(len(images))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, sub in images.groupby('cls'):
        idx = sub['idx'].to_numpy()
        s = feats[idx] @ feats[idx].T
        for a, b in zip(*np.where(np.triu(s, 1) >= thr)):
            ra, rb = find(idx[a]), find(idx[b])
            if ra != rb:
                parent[rb] = ra
    return np.array([find(i) for i in range(len(images))])


def task_similarity(name, task, images, feats):
    idx, y = C.task_members(images, task)
    f = feats[idx]
    k = len(task['classes'])
    cents = np.stack([f[y == c].mean(0) for c in range(k)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    return {'task': name, 'k': k, 'n': len(idx),
            'silhouette': float(silhouette_score(f, y, metric='cosine')),  # 낮을수록 클래스가 겹친다
            'centroid_cos': float((cents @ cents.T)[np.triu_indices(k, 1)].mean())}  # 높을수록 비슷하다


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--dup-thr', type=float, required=True,
                    help='같은 클래스 안에서 코사인 유사도가 이 값 이상이면 같은 물체로 묶는다')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--cap', type=int, default=400, help='재질 5클래스에서 클래스당 쓸 장수')
    ap.add_argument('--with-quads', action='store_true', help='4클래스 조합 5개도 과제에 넣는다')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    md = C.meta_dir(args.work_dir)
    images = pd.read_csv(md / 'images.csv')
    feats = np.load(md / 'feat_pretrained.npy')

    # 1) 중복 그룹
    images['group'] = pd.factorize(dup_groups(images, feats, args.dup_thr))[0]
    sizes = images['group'].value_counts()
    multi = sizes[sizes > 1]
    print(f'그룹 {len(sizes)}개, 2장 이상 묶인 그룹 {len(multi)}개({int(multi.sum())}장), '
          f'가장 큰 그룹 {int(sizes.max())}장')
    if sizes.max() > 20:
        print('경고: 너무 큰 그룹이 있습니다. --dup-thr를 높이는 것을 고려하세요.')

    # 2) 이미지별 fold 고정: 클래스 비율을 유지하고, 같은 그룹은 같은 fold에 둔다
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    images['fold'] = -1
    for f, (_, te) in enumerate(sgkf.split(images, images['label6'], images['group'])):
        images.loc[te, 'fold'] = f
    print('fold별 클래스 장수:')
    print(pd.crosstab(images['fold'], images['cls']))

    # 3) 재질 5클래스에서 클래스당 cap장을 고정 seed로 고른다 (부분집합 과제용)
    rng = np.random.default_rng(args.seed)
    images['cap'] = False
    for cls in C.MATERIALS:
        idx = images.index[images['cls'] == cls].to_numpy()
        images.loc[rng.choice(idx, min(args.cap, len(idx)), replace=False), 'cap'] = True
    images.to_csv(md / 'images.csv', index=False)

    # 4) 과제 목록과 과제별 유사도 (학습 없이 사전학습 특징으로 계산)
    tasks = C.build_tasks(args.with_quads)
    (md / 'tasks.json').write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding='utf-8')
    sim = pd.DataFrame([task_similarity(n, t, images, feats) for n, t in tasks.items()])
    sim.to_csv(md / 'similarity.csv', index=False)
    print(f'과제 {len(tasks)}개. silhouette가 낮을수록 클래스가 겹친다(어렵다):')
    print(sim.sort_values(['k', 'silhouette']).to_string(index=False, float_format='%.3f'))
    (md / 'folds_config.json').write_text(json.dumps(vars(args), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
