"""과제군 넓히기: trash가 낀 과제와 4·5클래스 과제 25개를 tasks.json과 similarity.csv에 더한다.

팀 분할 6분류에 계획이 맞지 않았던 것(예상이 2.7~4.4%p 낙관적)은 기존 21과제가 재질 부분집합뿐이라
trash가 낀 과제도, 5클래스를 넘는 과제도 없었기 때문으로 본다. 그래서 그 빈 곳을 채운다.
  재질 + trash 쌍 5개 (pair_xx-tr), 재질 2개 + trash 삼중 10개 (tri_xx-yy-tr),
  재질 4개 5개 (quad_..., trash 없음), 재질 4개 + trash 5개 (quint_...-tr)
재질은 기존과 같이 클래스당 cap장을, trash는 137장 전부를 쓴다 (pool 'cap_tr').
기존 과제, 이미지별 fold, 중복 그룹은 건드리지 않는다. 이미 있는 과제는 다시 쓰지 않는다.

사용: python add_tasks.py
"""
import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd

import common as C
from make_folds import task_similarity


def new_tasks():
    tasks = {}
    for k in (1, 2, 4):  # 재질 k개 + trash
        prefix = {1: 'pair', 2: 'tri', 4: 'quint'}[k]
        for combo in combinations(C.MATERIALS, k):
            name = prefix + '_' + '-'.join(C.ABBR[c] for c in combo) + '-tr'
            tasks[name] = {'classes': list(combo) + ['trash'], 'pool': 'cap_tr'}
    for combo in combinations(C.MATERIALS, 4):  # 재질 4개
        tasks['quad_' + '-'.join(C.ABBR[c] for c in combo)] = {'classes': list(combo), 'pool': 'cap'}
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    args = ap.parse_args()
    md = C.meta_dir(args.work_dir)
    images = pd.read_csv(md / 'images.csv')
    feats = np.load(md / 'feat_pretrained.npy')
    tasks = json.loads((md / 'tasks.json').read_text(encoding='utf-8'))
    sim = pd.read_csv(md / 'similarity.csv')
    add = {n: t for n, t in new_tasks().items() if n not in tasks}
    if add:
        tasks.update(add)
        (md / 'tasks.json').write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding='utf-8')
        rows = pd.DataFrame([task_similarity(n, t, images, feats) for n, t in add.items()])
        sim = pd.concat([sim[~sim.task.isin(add)], rows], ignore_index=True)
        sim.to_csv(md / 'similarity.csv', index=False)
    names = list(new_tasks())
    print(f'추가 {len(add)}개 (전체 과제 {len(tasks)}개). 새 과제의 유사도:')
    print(sim[sim.task.isin(names)].sort_values(['k', 'silhouette']).to_string(index=False, float_format='%.3f'))
    print(f'참고: all6 실루엣 {sim.set_index("task").silhouette["all6"]:.3f}')
    (md / 'tasks_ext.txt').write_text(','.join(names), encoding='utf-8')  # 대기열에서 --tasks로 쓴다


if __name__ == '__main__':
    main()
