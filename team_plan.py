"""팀 분할 6분류에서 '학습 전에 정한 프루닝 계획'이 맞는지 본다. 세 단계로 나눠, 예측을 학습 전에 파일로 남긴다.

  plan  : 22과제 중 all6을 뺀 21과제(fold 0~4, 파인튜닝 + movement, 2800 step)로 '실루엣 -> 비율별 정확도' 식을 만들고,
          팀 분할 학습 이미지의 실루엣으로 요구 85%(주)·88%·90%(참고)마다 고를 비율과 예상 정확도를 저장한다.
          all6은 팀 분할과 같은 이미지를 쓰는 과제라 식에서 뺀다 (넣으면 답을 미리 본 셈이다).
          고르는 방식은 plan_diag.py와 같다: 곡선(실루엣), 여유는 21과제 안에서 과제 하나씩 빼고 80%를 지키는 가장 작은 값.
  train : 팀 분할 원본(runs/team/seed0~2)을 movement로 균일하게 자르고 2800 step 회복한다.
          비율은 요구마다 고른 비율과 그 바로 위 비율이다. 모델은 팀원 양자화용으로 통째로 저장한다.
  judge : 미리 정한 판정을 낸다.
          예측 성공 : 고른 비율에서 시드 3개 평균 test 정확도가 요구치 이상
          최대였나  : 바로 위 비율의 시드 평균이 요구치 미만이면 '지킬 수 있는 가장 큰 비율을 골랐다'
          예측 오차 : 예상 정확도(여유를 빼기 전) - 실제 시드 평균

2판 (--family ext): 1판에서 예상이 2.7~4.4%p 낙관적이었던 것을 본 뒤, 과제군만 넓힌다 (add_tasks.py의 25과제를
          fold 0·1로 더함: trash가 낀 과제, 4·5클래스 과제). 식과 고르는 방식은 그대로다.
          1판 결과를 이미 봤으므로 판정은 새 팀 분할 원본(seed 3·4·5)으로 한다. 판정 기준(결과를 보기 전에 정함):
          요구 85%에서 예측 성공이고 |예측 오차| ≤ 1.5%p이면 '6분류에 적용됨'.

사용: python team_plan.py plan && python team_plan.py train && python team_plan.py judge > work/team_plan_result.txt
      python team_plan.py plan --family ext --plan-file team_plan2.json   # 2판 (train·judge도 같은 --plan-file,
                                                                            --seeds 3 4 5 --out-name team_plan2)
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

import common as C
import prune_common as P
from plan_diag import MAIN, calibrate, climb, fit, load_ft, load_ft_prune
from plan_eval import GRID, REC_STEPS, train_silhouettes

TARGETS = [85.0, 88.0, 90.0]  # 첫 값이 주
EXCLUDE = 'all6'
MAX_ERR = 1.5  # 2판 판정: 요구 85%에서 허용하는 예측 오차 (%p)


def plan(args, runs):
    tests = [load_ft(runs, [0, 1])[0], load_ft(runs, [2, 3, 4])[0]]
    if args.family == 'ext':  # 넓힌 과제(fold 0·1)는 prune_grid 결과에만 있다. 기존 과제는 비율이 모자라 저절로 빠진다
        tests.append(load_ft_prune(runs, [0, 1])[0])
    V = pd.concat(tests)
    V = V[(V.index.get_level_values('task') != EXCLUDE) & ~V.index.duplicated()]
    sil = train_silhouettes(args.work_dir, args.data_root, list(V.index))
    sim = pd.read_csv(C.meta_dir(args.work_dir) / 'similarity.csv').set_index('task')
    tt = V.groupby(level='task').mean()
    tt['sil'], tt['k'] = sil.groupby(level='task').mean(), sim.k

    images, tasks = C.load_meta(args.work_dir, args.data_root)
    idx, y = C.team_split(images)['train']
    feats = np.load(C.meta_dir(args.work_dir) / 'feat_pretrained.npy')
    s_team, k_team = float(silhouette_score(feats[idx], y, metric='cosine')), len(C.CLASSES6)
    trash = sum('trash' in tasks[t]['classes'] for t in tt.index)
    print(f'=== 팀 분할 6분류 계획 ({args.family}: {len(tt)}과제, 그중 trash 포함 {trash}개, 조건 {len(V)}개) ===')
    print('과제 클래스 수: ' + ', '.join(f'{k}클래스 {n}개' for k, n in tt.k.value_counts().sort_index().items()))
    print(f'팀 분할 학습 이미지 실루엣 {s_team:.3f} (식에 쓴 과제 범위 {tt.sil.min():.3f}~{tt.sil.max():.3f}, '
          f'all6 전체 이미지 {sim.silhouette[EXCLUDE]:.3f})')

    out = {'silhouette': s_team, 'family': args.family, 'n_tasks': len(tt), 'targets': {},
           'made': time.strftime('%Y-%m-%d %H:%M:%S')}
    for a in TARGETS:
        feas = V[0.0] >= a
        conds = {}
        for c in V.index[feas.to_numpy()]:
            conds.setdefault(c[0], []).append(c)
        m = calibrate(tt, conds, V, sil, MAIN, a)
        pred = fit(tt, MAIN)(s_team, k_team)
        g = climb(pred - m >= a)
        nxt = GRID[GRID.index(g) + 1] if g < GRID[-1] else None
        out['targets'][str(a)] = {'ratio': g, 'next': nxt, 'margin': float(m),
                                  'pred': dict(zip(map(str, GRID), map(float, pred)))}
        ref = {r: climb(fit(tt, r)(s_team, k_team) - calibrate(tt, conds, V, sil, r, a) >= a)
               for r in ['FCA식', '곡선(실루엣+k)']}
        print(f'\n  요구 {a:g}%: 고른 비율 {g:g} (바로 위 {nxt}), 여유 {m:.2f}%p'
              f'   참고: FCA식 {ref["FCA식"]:g}, 곡선(실루엣+k) {ref["곡선(실루엣+k)"]:g}')
        print('    예상 정확도: ' + ', '.join(f'{r:g} {p:.2f}' for r, p in zip(GRID, pred)))
    path = runs.parent / args.plan_file
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n저장: {path} (학습 전 예측)')


def train(args, runs):
    pl = json.loads((runs.parent / args.plan_file).read_text(encoding='utf-8'))
    ratios = sorted({x for t in pl['targets'].values() for x in (t['ratio'], t['next']) if x})
    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
    train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    cache = C.EvalCache(args.work_dir, args.device)
    print(f'자를 비율 {ratios}, 시드 {args.seeds}, 회복 {args.steps} step')
    for s in args.seeds:
        base_dir = runs / 'team' / f'seed{s}'
        base, move = P.load_base(base_dir, len(C.CLASSES6), args.device), P.load_movement(base_dir)
        for r in ratios:
            out = runs.parent / args.out_name / f'seed{s}' / f'mov_r{r}.json'
            if out.exists():
                continue
            t0 = time.time()
            C.set_seed(s)
            model = copy.deepcopy(base)
            P.prune(model, P.make_importance('mov_cum', model, move), r)
            P.recalibrate_bn(model, cache, tr_idx)
            P.recover(model, train_ds, args.steps, args.device, args.workers, s)
            st = P.model_stats(model)
            info = {'seed': s, 'ratio': r, 'steps': args.steps, 'split': 'team_split.json',
                    'val': C.metrics(C.predict_logits(model, cache, va_idx), va_y),
                    'test': C.metrics(C.predict_logits(model, cache, te_idx), te_y),
                    'params': st['params'], 'macs': st['macs'], 'time_sec': round(time.time() - t0, 1)}
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu(), out.with_suffix('.pt'))  # 팀원 양자화용 (구조째 저장, weights_only=False로 불러오기)
            out.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
            print(f"seed{s} 비율 {r}: test {100 * info['test']['acc']:.2f}%, params {st['params'] / 1e6:.2f}M "
                  f"({info['time_sec']}s)", flush=True)
            del model


def judge(args, runs):
    pl = json.loads((runs.parent / args.plan_file).read_text(encoding='utf-8'))
    acc = {}
    for s in args.seeds:
        acc[(s, 0.0)] = 100 * json.loads((runs / 'team' / f'seed{s}' / 'result.json').read_text(encoding='utf-8'))['test']['acc']
        for p in (runs.parent / args.out_name / f'seed{s}').glob('mov_r*.json'):
            j = json.loads(p.read_text(encoding='utf-8'))
            acc[(s, j['ratio'])] = 100 * j['test']['acc']
    tab = pd.Series(acc).unstack(0)  # 비율 x 시드
    print(f'=== 팀 분할 6분류 계획 판정 ({pl.get("family", "orig")}, 예측 저장 {pl["made"]}, '
          f'실루엣 {pl["silhouette"]:.3f}, 시드 {args.seeds}) ===')
    print('\n시드별 test 정확도 (%)')
    print(tab.assign(평균=tab.mean(axis=1), sd=tab.std(axis=1)).round(2).to_string())
    print('\n[판정] (미리 정함)')
    for a, t in pl['targets'].items():
        g, nxt = t['ratio'], t['next']
        real = tab.loc[g].mean() if g in tab.index else np.nan
        pred = t['pred'][str(g)]
        ok = real >= float(a)
        line = (f'  요구 {float(a):g}%{" (주)" if float(a) == TARGETS[0] else ""}: 고른 비율 {g:g}, '
                f'실제 {real:.2f} (예상 {pred:.2f}, 오차 {pred - real:+.2f}) -> 예측 {"성공" if ok else "실패"}')
        if nxt is None:
            line += ', 이미 최대 비율'
        elif nxt in tab.index:
            nreal = tab.loc[nxt].mean()
            line += f', 바로 위 비율 {nxt:g}: {nreal:.2f} -> {"가장 큰 비율을 골랐다" if nreal < float(a) else "더 자를 수 있었다"}'
        print(line)
        if float(a) == TARGETS[0] and pl.get('family') == 'ext':
            fine = ok and abs(pred - real) <= MAX_ERR
            print(f'  2판 판정 (요구 {float(a):g}%에서 성공이고 |오차| ≤ {MAX_ERR}%p): '
                  f'{"6분류에 적용됨" if fine else "6분류에 적용 안 됨"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['plan', 'train', 'judge'])
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=REC_STEPS)
    ap.add_argument('--out-name', default='team_plan', help='모델 폴더 이름 (work/ 아래, 시험 실행은 다른 이름으로)')
    ap.add_argument('--family', choices=['orig', 'ext'], default='orig', help='계획식에 쓸 과제군 (ext: add_tasks.py의 과제 포함)')
    ap.add_argument('--plan-file', default='team_plan.json', help='학습 전 예측을 저장·읽는 파일 (work/ 아래)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    runs = C.runs_dir(args.work_dir)
    {'plan': plan, 'train': train, 'judge': judge}[args.step](args, runs)


if __name__ == '__main__':
    main()
