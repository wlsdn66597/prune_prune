"""잡음 확인 실험(판단 관문). 진단(probe_diag_v1)에서 본 후보 간 차이가 회복 학습의 무작위성보다 큰지,
같은 계산량을 회복에 더 쓰는 것보다 나은지 확인한다.

  A. 같은 구조 시드 대조: 균일 후보를 회복 시드 0~7로 학습해 잡음만으로 생기는 '사후 최고'를 잰다
  B. 선택 후보 재학습: 진단에서 사후 최고 정확도, 사후 최저 NLL, prefix_280·prefix_56 NLL 규칙이 고른 후보를
     시드 0~2로 다시 학습하고 테스트셋까지 평가한다 (균일과 같은 시드끼리 짝)
  C. 계산량 맞춘 비교군: 균일 후보를 1232 / 1680 / 2800 step 회복 x 시드 0~2
     (후보 8개일 때 prefix_56 선택 = 7x56+840, SH 56>168>280 = 1680, prefix_280 선택 = 7x280+840 step)

진단 결과는 읽기만 한다. 후보는 진단과 같은 방법으로 다시 만들고(균일은 P.prune, 나머지는 저장된 cfg로
J.materialize) 해시로 같은 모델인지 확인한다. 회복 절차도 진단과 같다(BN 재보정 -> AdamW 1e-4, cosine,
같은 worker 수). 그래서 시드 0은 진단 실행의 재현이다. '시드'는 회복 학습의 시드(데이터 순서, 증강)이고
기준 모델과 분할은 진단과 같다.

사용: nohup bash run_noise_gate.sh > work/noise_gate_queue.log 2>&1 &   (대기 -> 시험 -> 본 실행 -> 요약)
      python noise_gate.py --wait-only            # 다른 학습이 끝날 때까지 기다리기만 함
      python noise_gate.py --smoke                # 입력·후보 해시 전체 확인 + 16 step 시험 학습
      python noise_gate.py --shard 0 --nshards 3  # 본 실행
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

TRAIN_SCRIPTS = {
    'baseline_team.py', 'train_base.py', 'prune_grid.py', 'stage_sensitivity.py', 'select_by_recovery.py',
    'trp_search.py', 'trp_joint.py', 'gradual_prune.py', 'gradual_alloc.py', 'make_team_trp.py',
    'train_mobile.py', 'make_temp_pruned.py', 'diagnose_trp_probe.py', 'noise_gate.py',
}
ROLES = ['oracle_acc', 'oracle_nll', 'pick_p280', 'pick_p56']
GPU_BUSY = 10  # GPU 사용률(%)이 이 이상이면 누군가 계산 중으로 본다


def log(msg):
    print(time.strftime('[%m-%d %H:%M:%S] ') + msg, flush=True)


# ---------------- 대기 ----------------

def own_processes():
    """이 사용자의 살아 있는 프로세스 (pid, argv). 자기 자신과 좀비는 뺀다."""
    me, uid = os.getpid(), os.getuid()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            if d.stat().st_uid != uid or (d / 'stat').read_text().rsplit(')', 1)[1].split()[0] in {'Z', 'X'}:
                continue
            argv = [a.decode(errors='replace') for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
        except (OSError, IndexError):
            continue
        if argv:
            yield d.name, argv


def is_workload(argv):
    """실험 스크립트(python ...py)나 다른 대기열(run_*.sh)인지. 이 대기열 자신과 대기만 하는 프로세스는 뺀다."""
    exe, names = Path(argv[0]).name, [Path(a).name for a in argv[1:]]
    if exe.startswith('python'):
        return any(n in TRAIN_SCRIPTS for n in names) and '--wait-only' not in argv
    if exe in {'bash', 'sh', 'dash'}:
        return any(n.startswith('run_') and n.endswith('.sh') and n != 'run_noise_gate.sh' for n in names)
    return False


def gpu_util_peak(samples=5):
    """GPU 사용률을 1초 간격으로 재서 최댓값을 낸다. 메모리만 잡고 쉬는 프로세스(열린 노트북 커널 등)는 0%다."""
    peak = 0
    for i in range(samples):
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=20, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        peak = max([peak] + [int(t) for t in out.split() if t.isdigit()])
        if i + 1 < samples:
            time.sleep(1)
    return peak


def wait_for_idle(poll):
    """아는 실험 스크립트와 run_*.sh 대기열이 없고 GPU도 쉬는 상태가 두 번 연속 확인되면 끝난다.

    이름을 모르는 학습도 GPU 사용률로 잡는다. 메모리만 잡은 채 쉬는 프로세스 때문에 끝없이 기다리지는 않는다.
    """
    if not Path('/proc').is_dir():
        raise SystemExit('대기 기능은 리눅스 서버에서만 동작합니다.')
    idle, last, warned = 0, None, False
    while idle < 2:
        busy = {pid: ' '.join(argv)[:150] for pid, argv in own_processes() if is_workload(argv)}
        util = gpu_util_peak()
        if util is None and not warned:
            log('nvidia-smi를 쓸 수 없어 실행 중인 스크립트 이름으로만 판단합니다.')
            warned = True
        if util is not None and util >= GPU_BUSY:
            busy['gpu'] = f'GPU 사용률 {util}%'
        if busy:
            idle = 0
            if sorted(busy) != last:
                log('기다리는 중: ' + json.dumps(busy, ensure_ascii=False))
                last = sorted(busy)
        else:
            idle += 1
            last = None
        if idle < 2:
            time.sleep(poll)
    log('다른 학습이 모두 끝났습니다.')


# ---------------- 실험 설계 (가벼움, 요약 스크립트도 씀) ----------------

def add_design_args(ap):
    """실험 설계 인자. summarize_noise_gate.py도 같은 기본값을 쓴다."""
    ap.add_argument('--a-seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7], help='균일 추가 시드 (A)')
    ap.add_argument('--b-seeds', type=int, nargs='+', default=[0, 1, 2], help='선택 후보와 균일의 시드 (B)')
    ap.add_argument('--c-seeds', type=int, nargs='+', default=[0, 1, 2], help='균일 장기 회복 시드 (C)')
    ap.add_argument('--c-steps', type=int, nargs='+', default=[1232, 1680, 2800], help='균일 장기 회복 step (C)')
    ap.add_argument('--final-steps', type=int, default=840)


def load_conditions(diag_root):
    """진단에서 후보가 모두 끝난 조건만 읽는다."""
    conds = []
    for man in sorted(Path(diag_root).glob('*/fold*_seed*/r*/candidates.json')):
        manifest = json.loads(man.read_text(encoding='utf-8'))
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(man.parent.glob('cand*/result.json'))]
        if not rows or len(rows) < len(manifest['candidates']):
            continue
        conds.append({'key': man.parent.relative_to(diag_root).as_posix(), 'manifest': manifest, 'rows': rows})
    return conds


def picks(rows):
    """진단(시드 0, 검증셋)이 고른 후보. 동률은 진단과 같이 NLL·정확도·이름 순으로 정한다."""
    final = str(rows[0]['spec']['final_steps'])
    f = lambda r: r['curve'][final]
    out = {'uniform': next(r for r in rows if 'uniform_mov' in r['aliases']),
           'oracle_acc': min(rows, key=lambda r: (-f(r)['acc'], f(r)['nll'], r['name'])),
           'oracle_nll': min(rows, key=lambda r: (f(r)['nll'], -f(r)['acc'], r['name']))}
    for t in ('280', '56'):
        if all(t in r['curve'] for r in rows):
            out[f'pick_p{t}'] = min(rows, key=lambda r: (r['curve'][t]['nll'], -r['curve'][t]['acc'], r['name']))
    return out


def build_jobs(conds, args):
    """(조건 번호, 후보 이름, 시드, step). 균일의 A·B·C 시드가 겹치면 한 번만 돈다."""
    jobs = set()
    for i, c in enumerate(conds):
        roles = picks(c['rows'])
        uni = roles['uniform']['name']
        jobs.update((i, uni, s, args.final_steps) for s in args.a_seeds)
        jobs.update((i, r['name'], s, args.final_steps) for r in roles.values() for s in args.b_seeds)
        jobs.update((i, uni, s, st) for st in args.c_steps for s in args.c_seeds)
    return sorted(jobs, key=lambda j: (-j[3], j[0], j[1], j[2]))  # 긴 실행부터 돌아가며 나눠 맡게


# ---------------- 학습 (무거운 import는 대기 후에) ----------------

def import_heavy():
    global torch, F, DataLoader, C, P, J
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import common as C
    import prune_common as P
    import trp_joint as J


def digest_file(path):  # 진단과 같은 방식
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model_digest(model):  # 진단과 같은 방식
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def condition_env(cond, args, shared):
    """조건별 기준 모델·분할을 한 번만 읽고, 진단 때와 같은 입력인지 확인한다."""
    key, memo = cond['key'], shared['memo']
    if key not in memo:
        spec = cond['manifest']['spec']
        task, fold, seed = spec['task'], spec['fold'], spec['seed']
        base_dir = C.runs_dir(args.work_dir) / 'base' / task / f'fold{fold}_seed{seed}'
        split = C.task_split(shared['images'], shared['tasks'][task], fold)
        (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
        inputs = spec['inputs']
        if (inputs['base'] != digest_file(base_dir / 'model.pt')
                or inputs['movement'] != digest_file(base_dir / 'movement.pt')
                or inputs['train_indices'] != tr_idx.tolist() or inputs['val_indices'] != va_idx.tolist()):
            raise SystemExit(f'{key}: 기준 모델이나 분할이 진단 때와 다릅니다.')
        workers = spec['workers'] if args.workers is None else args.workers
        memo[key] = {
            'spec': spec, 'tr_idx': tr_idx, 'te_idx': te_idx, 'te_y': te_y,
            'base': P.load_base(base_dir, len(shared['tasks'][task]['classes']), args.device),
            'move': P.load_movement(base_dir),
            'ctx': {'cache': shared['cache'], 'va_idx': va_idx, 'va_y': va_y, 'device': args.device,
                    'workers': workers,
                    'train_ds': C.ImageList(shared['images']['path'].to_numpy()[tr_idx], tr_y, C.train_tf)}}
    return memo[key]


def build_candidate(env, cand):
    """진단의 make_candidates와 같은 방법으로 후보를 다시 만든다."""
    if 'uniform_mov' in cand['aliases']:
        model = copy.deepcopy(env['base'])
        P.prune(model, P.make_importance('mov_cum', model, env['move']), env['spec']['ratio'])
        return model
    return J.materialize(env['base'], env['move'], cand['cfg'])


def val_metrics(model, ctx):
    return C.metrics(C.predict_logits(model, ctx['cache'], ctx['va_idx']), ctx['va_y'])


def train_curve(model, ctx, seed, steps, checkpoints):
    """진단의 train_curve와 같은 절차. 시드만 바꿀 수 있다."""
    C.set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ctx['train_ds'], C.BATCH, shuffle=True, num_workers=ctx['workers'], generator=g,
                        pin_memory=ctx['device'] != 'cpu', persistent_workers=ctx['workers'] > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    curve = {'0': val_metrics(model, ctx)}
    step = 0
    while step < steps:
        for x, y in loader:
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(ctx['device'], non_blocking=True)),
                                   y.to(ctx['device'], non_blocking=True))
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step in checkpoints or step == steps:
                curve[str(step)] = val_metrics(model, ctx)
            if step >= steps:
                break
    del loader
    model.eval()
    return curve


def run_job(cond, cand_name, seed, steps, args, shared, out_root, checkpoints):
    out = out_root / cond['key'] / cand_name / f'rs{seed}_st{steps}.json'
    if out.exists():
        return out, None
    env = condition_env(cond, args, shared)
    cand = next(r for r in cond['rows'] if r['name'] == cand_name)
    t0 = time.time()
    model = build_candidate(env, cand)
    same = model_digest(model) == cand['model_sha256']
    C.set_seed(seed)
    P.recalibrate_bn(model, shared['cache'], env['tr_idx'])
    curve = train_curve(model, env['ctx'], seed, steps, checkpoints)
    logits = C.predict_logits(model, shared['cache'], env['te_idx'])
    params, macs = J.resources(model)
    spec = env['spec']
    info = {'cond': cond['key'], 'task': spec['task'], 'fold': spec['fold'], 'base_seed': spec['seed'],
            'ratio': spec['ratio'], 'cand': cand_name, 'aliases': cand['aliases'], 'same_initial_model': same,
            'recovery_seed': seed, 'steps': steps, 'workers': env['ctx']['workers'], 'val_curve': curve,
            'test': C.metrics(logits, env['te_y']), 'test_pred': logits.argmax(1).tolist(),
            'params': params, 'macs': macs, 'params_util': params / spec['target_params'],
            'macs_util': macs / spec['target_macs'], 'time_sec': round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
    tmp.replace(out)  # 끝난 실행만 결과 파일이 생긴다
    return out, info


def smoke(conds, args, shared):
    """본 실행 전에 모든 조건의 입력과 필요한 후보 전부의 해시를 확인하고, 짧은 학습 하나로 저장까지 시험한다."""
    jobs = build_jobs(conds, args)
    needed = sorted({(i, name) for i, name, _, _ in jobs})
    bad = []
    for i, name in needed:
        env = condition_env(conds[i], args, shared)
        cand = next(r for r in conds[i]['rows'] if r['name'] == name)
        if model_digest(build_candidate(env, cand)) != cand['model_sha256']:
            bad.append(f"{conds[i]['key']} {name}")
    log(f'입력 확인: 조건 {len(conds)}개 모두 진단과 같음, 후보 {len(needed)}개 중 해시 불일치 {len(bad)}개')
    if bad:
        raise SystemExit('진단 때와 다른 후보: ' + ', '.join(bad)
                         + ' (common.py, prune_common.py, trp_joint.py가 바뀌었는지 확인하세요)')
    out_root = C.runs_dir(args.work_dir) / (args.out_name + '_smoke')
    shutil.rmtree(out_root, ignore_errors=True)
    uni = picks(conds[0]['rows'])['uniform']['name']
    _, info = run_job(conds[0], uni, 1, 16, args, shared, out_root, {8})
    log(f"시험 학습: {conds[0]['key']} {uni} 16 step, val acc={info['val_curve']['16']['acc']:.4f}, "
        f"test acc={info['test']['acc']:.4f} ({info['time_sec']}s)")
    log(f'시험 통과. 본 실행은 {len(jobs)}개, 총 {sum(j[3] for j in jobs)} step입니다.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--diag-name', default='probe_diag_v1')
    ap.add_argument('--out-name', default='noise_gate_v1')
    add_design_args(ap)
    ap.add_argument('--checkpoints', type=int, nargs='+', default=[56, 168, 280])
    ap.add_argument('--workers', type=int, default=None, help='기본은 진단과 같은 값 (증강 난수를 맞추려고)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--wait-only', action='store_true', help='다른 학습이 끝날 때까지 기다리고 종료')
    ap.add_argument('--poll-seconds', type=float, default=60)
    ap.add_argument('--smoke', action='store_true', help='입력·후보 해시 전체 확인 + 16 step 시험 학습')
    args = ap.parse_args()

    if args.wait_only:
        wait_for_idle(args.poll_seconds)
        return
    diag_root = Path(args.work_dir) / 'runs' / args.diag_name
    conds = load_conditions(diag_root)
    if not conds:
        raise SystemExit(f'끝난 진단 결과가 없습니다: {diag_root}')
    import_heavy()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA를 쓸 수 없습니다. 가상환경을 확인하세요.')
    images, tasks = C.load_meta(args.work_dir, args.data_root)
    shared = {'images': images, 'tasks': tasks, 'cache': C.EvalCache(args.work_dir, args.device), 'memo': {}}
    if args.smoke:
        smoke(conds, args, shared)
        return

    out_root = C.runs_dir(args.work_dir) / args.out_name
    jobs = build_jobs(conds, args)
    mine = jobs[args.shard::args.nshards]
    log(f'조건 {len(conds)}개, 전체 실행 {len(jobs)}개 중 이 프로세스 {len(mine)}개')
    checkpoints = set(args.checkpoints)
    left, ran, t0 = sum(j[3] for j in mine), 0, time.time()
    for n, (i, cand, seed, steps) in enumerate(mine, 1):
        _, info = run_job(conds[i], cand, seed, steps, args, shared, out_root, checkpoints)
        left -= steps
        if info is None:
            continue
        ran += steps
        last = info['val_curve'][str(steps)]
        log(f"[{n}/{len(mine)}] {conds[i]['key']} {cand} 시드{seed} {steps}step: val acc={last['acc']:.4f}, "
            f"test acc={info['test']['acc']:.4f}{'' if info['same_initial_model'] else ', 초기 모델 불일치!'} "
            f"({info['time_sec']}s, 남은 시간 약 {(time.time() - t0) / ran * left / 60:.0f}분)")
    log('끝')


if __name__ == '__main__':
    main()
