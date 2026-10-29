"""약한 클래스 보호 채널 선택의 확인과 분해 (새 팀 분할 원본 seed 6~10, 비율 0.4).

team_protect.py(seed 0~5)는 미리 정한 판정으로 '효과 없음'이었다(metal 0.4에서 0.00, 0.6에서 −2.42%p).
대신 결과를 본 뒤, 0.4에서 전체 정확도가 +1.97%p(6/6 시드) 오른 것을 찾았다. 또 metal이 안 오른 이유로
'손실 기울기 기반 중요도는 이미 잘 맞히는 클래스(metal 95%)에서 기울기가 거의 0이라 그 클래스의 채널을 못 찾는다'를 의심한다.
그래서 같은 시드를 다시 쓰지 않고 새 원본으로 네 가지를 비교한다 (값은 모두 고정, 자르기·BN 재보정·2800 step 회복은 같다).
  A mov      : 표준 movement
  B protect  : movement + 위험 가중 클래스별 손실 Taylor (team_protect.py와 같다)
  C balanced : movement + 클래스 균등 가중 클래스별 손실 Taylor (위험 가중을 뺀 것)
  D logit    : movement + 위험 가중 클래스별 정답 logit Taylor
판정 (결과를 보기 전에 정함, 시드 5개):
  재현     : B − A 전체 정확도 평균 +1.0%p 이상이고 5개 중 4개 이상의 시드에서 B가 나음
  원인     : B − C 전체 정확도 평균 +0.5%p 이상이면 '위험 가중이 필요', 아니면 '클래스별로 따로 잰 중요도가 원인'
  metal 보호: D − A 가장 약한 클래스(실루엣 최저) 재현율 +3%p 이상이고 전체 정확도 −0.5%p 이내

사용: python team_protect2.py train && python team_protect2.py judge > work/team_protect2_result.txt
      (먼저 python baseline_team.py --seeds 6 7 8 9 10)
"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import common as C
import prune_common as P
import team_weighted as TW
from plan_eval import REC_STEPS
from team_protect import class_taylor, protect_scores

ARMS = ['mov', 'protect', 'balanced', 'logit']
NAMES = {'mov': 'A 표준', 'protect': 'B 보호(손실)', 'balanced': 'C 균등(손실)', 'logit': 'D 보호(logit)'}


def train(args, work):
    _, w = TW.class_weights(args.work_dir, args.data_root)
    images, _ = C.load_meta(args.work_dir, args.data_root)
    split = C.team_split(images)
    (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
    train_ds = C.ImageList(images['path'].to_numpy()[tr_idx], tr_y, C.train_tf)
    cache = C.EvalCache(args.work_dir, args.device)
    runs = C.runs_dir(args.work_dir)
    for s in args.seeds:
        base_dir = runs / 'team' / f'seed{s}'
        base, move = P.load_base(base_dir, len(C.CLASSES6), args.device), P.load_movement(base_dir)
        taylor = {}
        for arm in args.arms:
            out = work / args.out_name / arm / f'seed{s}' / f'mov_r{args.ratio}.json'
            if out.exists():
                continue
            if arm != 'mov':
                target = 'logit' if arm == 'logit' else 'loss'
                if target not in taylor:
                    taylor[target] = class_taylor(base, train_ds, np.asarray(tr_y), args.device, args.workers, s, target)
                _, so, si = protect_scores(move, taylor[target], np.ones_like(w) if arm == 'balanced' else w)
            t0 = time.time()
            C.set_seed(s)
            model = copy.deepcopy(base)  # 중요도는 자를 모델(복사본)의 모듈 이름으로 만들어야 한다
            imp = P.make_importance('mov_cum', model, move) if arm == 'mov' else P.ChannelScoreImportance(model, so, si)
            P.prune(model, imp, args.ratio)
            P.recalibrate_bn(model, cache, tr_idx)
            P.recover(model, train_ds, args.steps, args.device, args.workers, s)
            info = {'seed': s, 'ratio': args.ratio, 'arm': arm, 'steps': args.steps,
                    'val': C.metrics(C.predict_logits(model, cache, va_idx), va_y),
                    'test': C.metrics(C.predict_logits(model, cache, te_idx), te_y), 'time_sec': round(time.time() - t0, 1)}
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu(), out.with_suffix('.pt'))
            out.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
            print(f"seed{s} {NAMES[arm]}: test {100 * info['test']['acc']:.2f}% ({info['time_sec']}s)", flush=True)
            del model


def judge(args, work):
    sil, _ = TW.class_weights(args.work_dir, args.data_root)
    weak = C.CLASSES6[int(np.argmin(sil))]
    images, _ = C.load_meta(args.work_dir, args.data_root)
    te_idx, te_y = C.team_split(images)['test']
    cache = C.EvalCache(args.work_dir, args.device)
    rows = []
    for s in args.seeds:
        for arm in args.arms:
            p = work / args.out_name / arm / f'seed{s}' / f'mov_r{args.ratio}.pt'
            if not p.exists():
                print(f'모델 없음, 건너뜀: {p}')
                continue
            m = torch.load(p, map_location=args.device, weights_only=False).eval()
            pred = C.predict_logits(m, cache, te_idx).argmax(1).numpy()
            row = {'seed': s, 'arm': arm, '전체': 100 * (pred == te_y).mean(),
                   'macroF1': 100 * f1_score(te_y, pred, average='macro')}
            row.update({c: 100 * (pred[te_y == i] == i).mean() for i, c in enumerate(C.CLASSES6)})
            rows.append(row)
    d = pd.DataFrame(rows)
    cols = ['전체', 'macroF1', *C.CLASSES6]
    print(f'=== 약한 클래스 보호 채널 선택: 확인과 분해 (팀 분할 seed {args.seeds}, 비율 {args.ratio:g}, test %) ===')
    print(f'가장 약한 클래스(실루엣 최저): {weak}\n')
    tab = d.groupby('arm')[cols].mean().reindex(args.arms).rename(index=NAMES)
    print(tab.round(2).to_string())
    t = d.pivot_table(index='seed', columns='arm', values=cols)

    def diff(c, a, b):
        x = (t[c][a] - t[c][b]).dropna()
        return x.mean(), int((x > 0).sum()), len(x)

    print()
    for a, b in [('protect', 'mov'), ('balanced', 'mov'), ('logit', 'mov'), ('protect', 'balanced')]:
        parts = [f'{c} {diff(c, a, b)[0]:+.2f} ({diff(c, a, b)[1]}/{diff(c, a, b)[2]})' for c in ['전체', 'macroF1', weak]]
        print(f'  {NAMES[a]} − {NAMES[b]}: ' + ', '.join(parts))
    rep_m, rep_n, rep_k = diff('전체', 'protect', 'mov')
    cause = diff('전체', 'protect', 'balanced')[0]
    lw, lo = diff(weak, 'logit', 'mov')[0], diff('전체', 'logit', 'mov')[0]
    print('\n[판정]')
    print(f'  재현 (B − A 전체 ≥ +1.0, 시드 4/5 이상): {"예" if rep_m >= 1.0 and rep_n >= 4 else "아니오"} '
          f'({rep_m:+.2f}, {rep_n}/{rep_k})')
    print(f'  원인 (B − C 전체 {cause:+.2f}): {"위험 가중이 필요" if cause >= 0.5 else "클래스별로 따로 잰 중요도가 원인"}')
    print(f'  metal 보호 (D − A {weak} {lw:+.2f}, 전체 {lo:+.2f}): {"예" if lw >= 3 and lo >= -0.5 else "아니오"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['train', 'judge'])
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=[6, 7, 8, 9, 10])
    ap.add_argument('--arms', nargs='+', default=ARMS, choices=ARMS)
    ap.add_argument('--ratio', type=float, default=0.4)
    ap.add_argument('--steps', type=int, default=REC_STEPS)
    ap.add_argument('--out-name', default='team_protect2', help='모델 폴더 (work/ 아래)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    pd.set_option('display.width', 200)
    work = C.runs_dir(args.work_dir).parent
    {'train': train, 'judge': judge}[args.step](args, work)


if __name__ == '__main__':
    main()
