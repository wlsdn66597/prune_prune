"""추론 지연시간: 자른 모델의 MACs 감소가 실제 CPU 속도로 얼마나 이어지는지 잰다.

서버에서 구조를 만들어 TorchScript로 저장하고(--export), 같은 파일을 서버 CPU와 RPi에서 잰다.
가중치 값은 속도와 무관하므로 균일 프루닝 구조는 무작위 가중치로 만든다. 기성 소형 모델은 속도 비교용이다.

사용: python measure_latency.py --export                          # 서버: work/latency_models/에 저장
      python measure_latency.py --threads 1 4                     # 서버 CPU에서 측정 (다른 실험이 없을 때)
      python measure_latency.py --dir latency_models --threads 1 4 # RPi: 폴더와 이 파일만 복사, torch만 있으면 됨
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

RATIOS = [0.4, 0.6, 0.8, 0.9, 0.95]


def build_models(work):
    import torchvision
    import common as C
    import prune_common as P
    models = {'resnet18': C.build_model(6, pretrained=False)}
    for r in RATIOS:
        m = C.build_model(6, pretrained=False)
        P.prune(m, P.make_importance('l2', m, None), r)
        models[f'resnet18_u{r}'] = m
    for name in ('trp_bn_b0.16', 'trp_bn_b0.04'):  # 팀 모델 (make_team_trp.py)
        path = Path(work) / 'team_trp' / f'{name}.pt'
        if path.exists():
            models[f'resnet18_{name}'] = torch.load(path, map_location='cpu', weights_only=False)
    models['mnv3_small'] = torchvision.models.mobilenet_v3_small(num_classes=6)
    models['mnv2_x0.5'] = torchvision.models.mobilenet_v2(num_classes=6, width_mult=0.5)
    models['mnv2'] = torchvision.models.mobilenet_v2(num_classes=6)
    return models


def export(work, out_dir):
    import torch_pruning as tp
    out_dir.mkdir(parents=True, exist_ok=True)
    ex, info = torch.randn(1, 3, 224, 224), {}
    for name, m in build_models(work).items():
        m = m.cpu().eval()
        macs, params = tp.utils.count_ops_and_params(m, ex)
        with torch.no_grad():
            torch.jit.trace(m, ex).save(str(out_dir / f'{name}.pt'))
        info[name] = {'params': int(params), 'macs': int(macs)}
        print(f'{name:<24} params {params / 1e6:7.3f}M  MACs {macs / 1e6:8.1f}M', flush=True)
    (out_dir / 'models.json').write_text(json.dumps(info, indent=1), encoding='utf-8')


def latency(path, threads, warmup, runs, optimize):
    torch.set_num_threads(threads)
    m = torch.jit.load(str(path), map_location='cpu').eval()
    if optimize:
        try:  # conv-BN 접기 등 배포 때 하는 최적화. 플랫폼에서 안 되면 그대로 잰다
            m = torch.jit.optimize_for_inference(torch.jit.freeze(m))
        except Exception:
            pass
    x = torch.randn(1, 3, 224, 224)
    ts = []
    with torch.inference_mode():
        for _ in range(warmup):
            m(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--dir', default=None, help='저장된 모델 폴더 (기본: work/latency_models)')
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--no-optimize', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.dir) if args.dir else Path(args.work_dir) / 'latency_models'
    if args.export:
        export(args.work_dir, out_dir)
        return

    info = json.loads((out_dir / 'models.json').read_text(encoding='utf-8'))
    ref = info['resnet18']
    lat = {n: {t: latency(out_dir / f'{n}.pt', t, args.warmup, args.runs, not args.no_optimize) for t in args.threads}
           for n in info}
    host = platform.node() or 'host'  # 기기마다 따로 저장 (요약 스크립트가 읽는다)
    (out_dir / f'latency_{host}.json').write_text(json.dumps(
        {'torch': torch.__version__, 'ms': {n: {str(t): v for t, v in d.items()} for n, d in lat.items()}},
        indent=1), encoding='utf-8')
    print(f'torch {torch.__version__}, 배치 1, 224x224, 중앙값 ms (속도 향상은 resnet18 대비)')
    head = ''.join(f'{f"{t}thr ms":>10}{"x":>7}' for t in args.threads)
    print(f'{"model":<24}{"params(M)":>10}{"MACs(M)":>9}{"MACs 비":>9}{head}')
    for n, v in info.items():
        cells = ''.join(f'{lat[n][t]:>10.2f}{lat["resnet18"][t] / lat[n][t]:>7.1f}' for t in args.threads)
        print(f'{n:<24}{v["params"] / 1e6:>10.3f}{v["macs"] / 1e6:>9.1f}{ref["macs"] / v["macs"]:>8.1f}x{cells}')


if __name__ == '__main__':
    main()
