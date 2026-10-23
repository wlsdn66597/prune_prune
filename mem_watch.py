"""실험 프로세스별 메모리 최고치를 기록한다 (리눅스).

GPU: nvidia-smi의 프로세스별 사용량. PyTorch는 한 번 잡은 GPU 메모리를 돌려주지 않아서 순간 최고치도 남는다.
RAM: /proc의 VmHWM(프로세스별 최고 RSS)과 PSS(여러 프로세스가 공유하는 메모리를 나눠서 센 실제 사용량).
DataLoader 워커는 부모와 메모리를 공유해서 RSS가 크게 보이므로, 실제 합계는 PSS로 본다.

사용: nohup python mem_watch.py > /dev/null 2>&1 &   # 2초마다 기록. 실험이 30분 동안 안 보이면 스스로 끝난다
      python mem_watch.py --report                    # 지금까지의 최고치
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

SCRIPTS = ['trp_search.py', 'gradual_prune.py', 'prune_grid.py', 'stage_sensitivity.py', 'train_base.py']
OUT = Path('work/mem_peak.json')


def read(path):
    try:
        return Path(path).read_text(errors='ignore')
    except OSError:
        return ''


def field(text, key):
    for line in text.splitlines():
        if line.startswith(key + ':'):
            return int(line.split()[1])
    return 0


def script_of(pid):
    cmd = read(f'/proc/{pid}/cmdline').replace('\0', ' ')
    return next((s for s in SCRIPTS if s in cmd), None)


def nvidia(query):
    try:
        return subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ''


def gpu_by_pid():
    res = {}
    for line in nvidia('--query-compute-apps=pid,used_memory').splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) == 2 and p[0].isdigit():
            try:
                res[int(p[0])] = float(p[1])
            except ValueError:
                pass
    return res


def gpu_total():
    try:
        return sum(float(x) for x in nvidia('--query-gpu=memory.used').split())
    except ValueError:
        return 0.0


def save(state):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding='utf-8')


def watch(interval, idle_minutes):
    state = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {'procs': {}, 'system': {}}
    seen_any, last_seen, n = False, time.time(), 0
    while True:
        found = {int(p): s for p in os.listdir('/proc') if p.isdigit() for s in [script_of(p)] if s}
        if found:
            gpu = gpu_by_pid()
            pss_total = 0
            now = time.strftime('%m-%d %H:%M:%S')
            for pid, s in found.items():
                status = read(f'/proc/{pid}/status')
                pss = field(read(f'/proc/{pid}/smaps_rollup'), 'Pss')
                pss_total += pss
                r = state['procs'].setdefault(str(pid), {'script': s, 'gpu_mib': 0.0, 'hwm_mb': 0.0, 'pss_mb': 0.0,
                                                         'first': now})
                r['role'] = 'worker' if found.get(field(status, 'PPid')) == s else 'main'
                r['gpu_mib'] = max(r['gpu_mib'], gpu.get(pid, 0.0))
                r['hwm_mb'] = max(r['hwm_mb'], field(status, 'VmHWM') / 1024)
                r['pss_mb'] = max(r['pss_mb'], pss / 1024)
                r['last'] = now
            group = '+'.join(sorted(set(found.values())))
            g = state['system'].setdefault(group, {'gpu_total_mib': 0.0, 'pss_total_mb': 0.0,
                                                   'mem_available_min_mb': 1e12})
            g['gpu_total_mib'] = max(g['gpu_total_mib'], gpu_total())
            g['pss_total_mb'] = max(g['pss_total_mb'], pss_total / 1024)
            g['mem_available_min_mb'] = min(g['mem_available_min_mb'], field(read('/proc/meminfo'), 'MemAvailable') / 1024)
            seen_any, last_seen = True, time.time()
        n += 1
        if n % 15 == 0:
            save(state)
        if seen_any and time.time() - last_seen > idle_minutes * 60:
            save(state)
            return
        time.sleep(interval)


def report():
    if not OUT.exists():
        raise SystemExit(f'{OUT}가 없습니다. 먼저 mem_watch.py를 백그라운드로 실행하세요.')
    state = json.loads(OUT.read_text(encoding='utf-8'))
    procs = list(state['procs'].values())
    for s in sorted({p['script'] for p in procs}):
        for role, name in (('main', '메인 프로세스'), ('worker', 'DataLoader 워커')):
            g = [p for p in procs if p['script'] == s and p.get('role') == role]
            if not g:
                continue
            line = f'{s:22s} {name} {len(g):3d}개: '
            if role == 'main':
                line += f"GPU 최고 {max(p['gpu_mib'] for p in g):6.0f} MiB, "
            line += (f"RAM 최고(RSS) {max(p['hwm_mb'] for p in g):6.0f} MB, "
                     f"실제 사용(PSS) 최고 {max(p['pss_mb'] for p in g):6.0f} MB")
            print(line)
    print('\n== 동시에 돈 프로세스 전체 ==')
    for group, g in state['system'].items():
        print(f"{group}: GPU 합계 최고 {g['gpu_total_mib']:.0f} MiB, 실험 프로세스 실제 RAM 합계(PSS) 최고 "
              f"{g['pss_total_mb']:.0f} MB, 시스템 여유 RAM 최저 {g['mem_available_min_mb']:.0f} MB")
    if procs and max(p['gpu_mib'] for p in procs) == 0:
        print('\n참고: 프로세스별 GPU 사용량이 0이면 컨테이너 안이라 nvidia-smi가 프로세스를 못 보는 경우예요. '
              'GPU 합계 값을 보세요.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--idle-minutes', type=float, default=30.0)
    args = ap.parse_args()
    if args.report:
        report()
    else:
        watch(args.interval, args.idle_minutes)


if __name__ == '__main__':
    main()
