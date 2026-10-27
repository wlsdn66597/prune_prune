"""다른 실험이 끝날 때까지 기다린다. 대기열 스크립트(run_*.sh)가 실험을 시작하기 전에 부른다.

아래 셋이 모두 없는 상태가 두 번 연속 확인되면 끝난다.
  - 이 사용자의 실험 스크립트 (python 스크립트.py 형태, TRAIN_SCRIPTS)
  - 다른 run_*.sh 대기열. 자기를 부른 대기열과, 나보다 늦게 기다리기 시작한 대기열은 뺀다 (먼저 온 순서로 돈다)
  - GPU 사용률 10% 이상. 이름을 모르는 학습도 잡고, 메모리만 잡고 쉬는 프로세스는 무시한다

사용: python wait_idle.py [--poll-seconds 60]
대기를 건너뛰려면 다른 터미널에서: pkill -f wait_idle.py
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

TRAIN_SCRIPTS = {
    'baseline_team.py', 'train_base.py', 'prune_grid.py', 'stage_sensitivity.py', 'select_by_recovery.py',
    'trp_search.py', 'trp_joint.py', 'gradual_prune.py', 'gradual_alloc.py', 'make_team_trp.py',
    'train_mobile.py', 'make_temp_pruned.py', 'diagnose_trp_probe.py', 'noise_gate.py',
}
GPU_BUSY = 10  # GPU 사용률(%)이 이 이상이면 누군가 계산 중으로 본다


def log(msg):
    print(time.strftime('[%m-%d %H:%M:%S] ') + msg, flush=True)


def processes():
    """이 사용자의 살아 있는 프로세스 {pid: (ppid, 시작 시각, argv)}. 좀비는 뺀다."""
    out, uid = {}, os.getuid()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            fields = (d / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = [a.decode(errors='replace') for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
        except (OSError, IndexError):
            continue
        if fields[0] not in {'Z', 'X'} and argv:
            out[int(d.name)] = (int(fields[1]), int(fields[19]), argv)
    return out


def is_waiter(argv):
    """기다리기만 하는 프로세스 (이 스크립트, noise_gate.py --wait-only)."""
    return any(Path(a).name == 'wait_idle.py' for a in argv) or '--wait-only' in argv


def busy_processes(procs, me):
    """기다려야 할 프로세스 {pid: 명령}."""
    parent = procs[me][0]
    mine = (procs[parent][1], parent) if parent in procs else (procs[me][1], me)
    waiting = {procs[p][0] for p, (_, _, argv) in procs.items() if is_waiter(argv)}  # 기다리는 중인 대기열
    busy = {}
    for pid, (_, start, argv) in procs.items():
        if pid in (me, parent) or is_waiter(argv):
            continue
        exe, names = Path(argv[0]).name, [Path(a).name for a in argv[1:]]
        if exe.startswith('python'):
            hit = any(n in TRAIN_SCRIPTS for n in names)
        elif exe in {'bash', 'sh', 'dash'}:
            # 나보다 늦게 기다리기 시작한 대기열은 나를 기다리므로 세지 않는다 (서로 기다리다 멈추는 것 방지)
            hit = (any(n.startswith('run_') and n.endswith('.sh') for n in names)
                   and not (pid in waiting and (start, pid) > mine))
        else:
            hit = False
        if hit:
            busy[str(pid)] = ' '.join(argv)[:150]
    return busy


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
    if not Path('/proc').is_dir():
        raise SystemExit('대기 기능은 리눅스 서버에서만 동작합니다.')
    idle, last, warned = 0, None, False
    while idle < 2:
        busy = busy_processes(processes(), os.getpid())
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--poll-seconds', type=float, default=60)
    wait_for_idle(ap.parse_args().poll_seconds)
