"""잡음 확인 실험(noise_gate.py) 요약. 표준 라이브러리만 쓴다.

판정 규칙 (결과를 보기 전에 정함):
  1) 진단의 사후 최고 후보(oracle_acc)가 테스트셋에서 균일보다 평균 +0.5%p 이상 높고,
     조건별 평균 차이에 대한 부호 뒤집기 검정 p < 0.1
  2) 실제로 쓸 수 있는 선택 규칙(prefix_56 또는 prefix_280 NLL) 중 하나 이상이,
     같은 계산량을 균일 회복에 더 쓴 것보다 테스트셋 평균이 높음
  둘 다 만족 -> 통과: 짧은 회복 곡선으로 후보를 고르는 방법(prefix-SH + TSE)을 이어간다
  아니면 -> 실패: 후보 간 차이는 잡음 수준. 같은 너비에서 채널 선택 연구로 돌리거나 분석 결과로 정리한다

사용: python summarize_noise_gate.py
"""
import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from noise_gate import ROLES, add_design_args, build_jobs, load_conditions, picks


def load_results(root):
    res = {}
    for p in Path(root).glob('*/fold*_seed*/r*/cand*/rs*_st*.json'):
        r = json.loads(p.read_text(encoding='utf-8'))
        res[r['cond'], r['cand'], r['recovery_seed'], r['steps']] = r
    return res


def acc(r, split):
    return 100 * (r['test']['acc'] if split == 'test' else r['val_curve'][str(r['steps'])]['acc'])


def paired(res, key, a, b, seeds, split, steps_a, steps_b, ref_seeds=None):
    """(a - b) 정확도 차이(%p). 결과가 없으면 None.

    ref_seeds가 없으면 같은 시드끼리 짝지은 차이의 평균, 있으면 a의 seeds 평균 - b의 ref_seeds 평균.
    같은 시드로 다시 돌려도 GPU 비결정성 때문에 결과가 달라지므로, 짝짓기보다 기준 시드를 많이 쓰는 쪽이 안정적이다.
    """
    if a == b and steps_a == steps_b:
        return 0.0
    if ref_seeds is None:
        d = [acc(res[key, a, s, steps_a], split) - acc(res[key, b, s, steps_b], split)
             for s in seeds if (key, a, s, steps_a) in res and (key, b, s, steps_b) in res]
        return sum(d) / len(d) if d else None
    xa = [acc(res[key, a, s, steps_a], split) for s in seeds if (key, a, s, steps_a) in res]
    xb = [acc(res[key, b, s, steps_b], split) for s in ref_seeds if (key, b, s, steps_b) in res]
    return mean(xa) - mean(xb) if xa and xb else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def signflip_p(xs):
    """조건별 차이의 평균이 0인지에 대한 양측 부호 뒤집기 검정 (정확 계산)."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    obs = abs(sum(xs)) - 1e-9
    hits = sum(abs(sum(s * x for s, x in zip(signs, xs))) >= obs
               for signs in itertools.product((1, -1), repeat=len(xs)))
    return hits / 2 ** len(xs)


def wtl(xs):
    xs = [x for x in xs if x is not None]
    w, l = sum(x > 1e-9 for x in xs), sum(x < -1e-9 for x in xs)
    return f'{w}/{len(xs) - w - l}/{l}'


def expected_max_normal(k):
    """표준정규 k개 중 최댓값의 기댓값 (k=8이면 약 1.42)."""
    h = 0.002
    return h * sum(x * k * math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
                   * (0.5 * (1 + math.erf(x / math.sqrt(2)))) ** (k - 1)
                   for x in (i * h for i in range(-5000, 5001)))


def expected_max_subset(vals, k):
    """값들 중 k개를 무작위로 뽑았을 때 최댓값의 기댓값 (정확 계산)."""
    v, n = sorted(vals), len(vals)
    if k > n:
        return None
    return sum(v[i] * math.comb(i, k - 1) for i in range(k - 1, n)) / math.comb(n, k)


def fmt(v, sign=True):
    return '-' if v is None else (f'{v:+.2f}' if sign else f'{v:.2f}')


def label(key):
    task, fold, ratio = key.split('/')
    return f"{task} f{fold.split('_')[0][4:]} {ratio}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--diag-name', default='probe_diag_v1')
    ap.add_argument('--out-name', default='noise_gate_v1')
    add_design_args(ap)
    ap.add_argument('--ref', choices=['paired', 'all'], default='paired',
                    help='B·C의 균일 기준: 같은 시드끼리 짝(paired) 또는 균일 시드 전체 평균(all)')
    args = ap.parse_args()
    ref = sorted(set(args.a_seeds) | set(args.b_seeds)) if args.ref == 'all' else None
    ref_txt = f'균일 시드 {ref} 평균 대비' if ref else '균일 대비, 같은 시드끼리 짝'
    runs = Path(args.work_dir) / 'runs'
    conds = load_conditions(runs / args.diag_name)
    res = load_results(runs / args.out_name)
    fin = args.final_steps
    jobs = build_jobs(conds, args)
    missing = sum((conds[i]['key'], c, s, t) not in res for i, c, s, t in jobs)
    print(f'=== 잡음 확인 실험 요약: 조건 {len(conds)}개, 끝난 실행 {len(jobs) - missing}/{len(jobs)}개 ===')
    if missing:
        print('  아직 끝나지 않은 실행이 있어 아래 값은 임시입니다.')

    diffs = [abs(acc(res[c['key'], r['name'], 0, fin], 'val') - 100 * r['curve'][str(fin)]['acc'])
             for c in conds for r in c['rows'] if (c['key'], r['name'], 0, fin) in res]
    if diffs:
        print(f'\n[재현] 시드 0 재실행 vs 진단 검증 정확도: {len(diffs)}개 중 같은 값 {sum(d < 1e-6 for d in diffs)}개, '
              f'최대 차이 {max(diffs):.2f}%p')
    bad = sorted({f"{r['cond']} {r['cand']}" for r in res.values() if not r['same_initial_model']})
    if bad:
        print('  주의: 진단 때와 초기 모델이 다른 후보 ' + ', '.join(bad))

    # ---- A: 같은 구조를 시드만 바꿔 학습 ----
    print('\n[A] 균일 구조를 시드만 바꿔 학습: 잡음만으로 생기는 차이 (정확도 %, 차이 %p)')
    print(f"  {'condition':<20}{'seeds':>6}{'val':>8}{'sd':>6}{'test':>8}{'sd':>6}"
          f"{'noise gap':>11}{'normal':>8}{'diag gap':>10}")
    rows_a = []
    for c in conds:
        key, uni, k = c['key'], picks(c['rows'])['uniform'], len(c['rows'])
        seeds = sorted(s for kk, cc, s, t in res if kk == key and cc == uni['name'] and t == fin)
        if len(seeds) < 2:
            print(f'  {label(key):<20}{len(seeds):>6}  (시드 부족)')
            continue
        val = [acc(res[key, uni['name'], s, fin], 'val') for s in seeds]
        test = [acc(res[key, uni['name'], s, fin], 'test') for s in seeds]
        sd_v, sd_t = statistics.stdev(val), statistics.stdev(test)
        top = expected_max_subset(val, k)
        noise = None if top is None else top - statistics.mean(val)
        normal = expected_max_normal(k) * sd_v
        diag = 100 * (max(r['curve'][str(fin)]['acc'] for r in c['rows']) - uni['curve'][str(fin)]['acc'])
        rows_a.append((sd_v, sd_t, noise, normal, diag))
        print(f'  {label(key):<20}{len(seeds):>6}{statistics.mean(val):>8.2f}{sd_v:>6.2f}'
              f'{statistics.mean(test):>8.2f}{sd_t:>6.2f}{fmt(noise):>11}{fmt(normal):>8}{fmt(diag):>10}')
    if rows_a:
        m = [mean(col) for col in zip(*rows_a)]
        print(f"  {'mean':<20}{'':>6}{'':>8}{m[0]:>6.2f}{'':>8}{m[1]:>6.2f}"
              f'{fmt(m[2]):>11}{fmt(m[3]):>8}{fmt(m[4]):>10}')
    print('  noise gap: 균일 시드 결과 중 후보 수만큼 뽑았을 때 최댓값 - 평균 (구조 차이 없이 잡음만으로 생기는 사후 최고)')
    print('  normal: 같은 값을 정규분포로 근사 (후보 8개면 1.42 x sd), diag gap: 진단의 사후 최고 - 균일 (시드 0)')
    print('  진단 후보들은 같은 시드를 공유해 잡음이 서로 상관돼 있으므로 실제 잡음 기대값은 이보다 조금 작을 수 있다.')

    # ---- B: 진단이 고른 후보를 다시 학습 ----
    new = [s for s in args.b_seeds if s != 0]
    per = {}
    for c in conds:
        p, key = picks(c['rows']), c['key']
        u = p['uniform']
        for role in ROLES:
            if role in p:
                a = p[role]['name']
                per[key, role] = {
                    'cand': a,
                    'diag': 100 * (p[role]['curve'][str(fin)]['acc'] - u['curve'][str(fin)]['acc']),
                    'val': paired(res, key, a, u['name'], new, 'val', fin, fin, ref),
                    'test': paired(res, key, a, u['name'], args.b_seeds, 'test', fin, fin, ref),
                    'test_new': paired(res, key, a, u['name'], new, 'test', fin, fin, ref)}
    col = lambda role, name: [v[name] for (_, r), v in per.items() if r == role]
    print(f'\n[B] 진단이 고른 후보를 다시 학습 (%p, {ref_txt}, 조건 평균)')
    print(f"  {'role':<12}{'diag val':>10}{'new val':>9}{'test':>8}{'new test':>10}{'test W/T/L':>12}{'p':>7}")
    for role in ROLES:
        print(f"  {role:<12}{fmt(mean(col(role, 'diag'))):>10}{fmt(mean(col(role, 'val'))):>9}"
              f"{fmt(mean(col(role, 'test'))):>8}{fmt(mean(col(role, 'test_new'))):>10}"
              f"{wtl(col(role, 'test')):>12}{fmt(signflip_p(col(role, 'test')), False):>7}")
    print(f'  diag val: 진단에서 고를 때 쓴 값(시드 0), new: 새 시드 {new}, test: 시드 {args.b_seeds}, '
          'p: test 조건 평균의 부호 뒤집기 검정')
    print('\n  조건별 test 차이 (%p, 괄호는 고른 후보, u는 균일과 같은 후보)')
    print(f"  {'condition':<20}" + ''.join(f'{r:>15}' for r in ROLES))
    for c in conds:
        cells = []
        for role in ROLES:
            v = per.get((c['key'], role))
            tag = '' if v is None else ('u' if v['cand'] == picks(c['rows'])['uniform']['name'] else v['cand'][4:])
            cells.append('-' if v is None else f"{fmt(v['test'])} ({tag})")
        print(f"  {label(c['key']):<20}" + ''.join(f'{x:>15}' for x in cells))

    # ---- C: 같은 계산량을 회복에 더 쓰면 ----
    gain = {}
    for c in conds:
        u = picks(c['rows'])['uniform']['name']
        for steps in args.c_steps:
            gain[c['key'], steps] = paired(res, c['key'], u, u, args.c_seeds, 'test', steps, fin, ref)
    print(f'\n[C] 균일 회복을 늘린 효과 (test, 시드 {args.c_seeds}, {fin} step {ref_txt} %p)')
    print(f"  {'condition':<20}" + ''.join(f'{f"{s}step":>10}' for s in args.c_steps))
    for c in conds:
        print(f"  {label(c['key']):<20}" + ''.join(f"{fmt(gain[c['key'], s]):>10}" for s in args.c_steps))
    print(f"  {'mean':<20}" + ''.join(f'{fmt(mean([gain[c["key"], s] for c in conds])):>10}' for s in args.c_steps))

    print('\n  같은 계산량 비교: 선택 규칙 - 회복 연장 (test %p, 조건 평균)')
    print(f'  선택 규칙 비용 = (후보 수 - 1) x 탐침 step + {fin} (탐침을 이어서 끝까지 학습할 때)')
    matched = {}
    for role, probe in (('pick_p56', 56), ('pick_p280', 280)):
        d, used = [], set()
        for c in conds:
            if (c['key'], role) not in per:
                continue
            cost = (len(c['rows']) - 1) * probe + fin
            steps = min(args.c_steps, key=lambda s: abs(s - cost))
            used.add(steps)
            s, g = per[c['key'], role]['test'], gain[c['key'], steps]
            d.append(None if s is None or g is None else s - g)
        matched[role] = mean(d)
        print(f'  {role:<10} vs 균일 {sorted(used)} step: {fmt(matched[role])} '
              f'(선택 승/무/패 {wtl(d)}, p={fmt(signflip_p(d), False)})')

    # ---- 판정 ----
    oracle = col('oracle_acc', 'test')
    g1 = mean(oracle) is not None and mean(oracle) >= 0.5 and (signflip_p(oracle) or 1) < 0.1
    g2 = any(v is not None and v > 0 for v in matched.values())
    print('\n[판정] 규칙은 이 파일 맨 위 설명에 있음')
    print(f"  1) 사후 최고 후보의 test 이득 {fmt(mean(oracle))}%p, p={fmt(signflip_p(oracle), False)}"
          f" -> {'충족' if g1 else '미충족'}")
    print('  2) 선택 규칙 - 같은 계산량 회복 연장: '
          + ', '.join(f'{r} {fmt(v)}' for r, v in matched.items()) + f" -> {'충족' if g2 else '미충족'}")
    if mean(oracle) is None or all(v is None for v in matched.values()):
        print('  => 판정 보류: 결과가 부족합니다')
    else:
        print('  => ' + ('(임시) ' if missing else '')
              + ('통과: prefix-SH + TSE 선택 연구로 진행' if g1 and g2 else
                 '실패: 후보 간 차이는 잡음 수준. 같은 너비 채널 선택 연구나 분석 결과로 정리'))


if __name__ == '__main__':
    main()
