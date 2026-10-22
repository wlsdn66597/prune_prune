"""프루닝 연구 공통 코드.

전처리와 학습 설정은 compare_fc.ipynb의 B 모델(fc를 6클래스로 교체)과 같다.
"""
import json
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---- compare_fc.ipynb와 같은 설정 ----
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH, LR = 32, 1e-4
TEAM_SPLIT_SEED = 42

CLASSES6 = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # ImageFolder 순서
MATERIALS = CLASSES6[:5]  # trash 제외
ABBR = {'cardboard': 'cb', 'glass': 'gl', 'metal': 'me', 'paper': 'pa', 'plastic': 'pl', 'trash': 'tr'}

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# 노트북의 eval_tf는 eval_pre 뒤에 ToTensor·Normalize를 붙인 것이다.
# 평가 이미지는 eval_pre까지 한 번만 적용해 캐시하고, 나머지(/255, 정규화)는 GPU에서 계산한다.
eval_pre = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])


def meta_dir(work):
    return Path(work) / 'meta'


def runs_dir(work):
    return Path(work) / 'runs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_meta(work, data_root=None):
    """images.csv(이미지 목록, fold 등)와 tasks.json을 읽고 이미지 절대 경로 열을 붙인다."""
    md = meta_dir(work)
    images = pd.read_csv(md / 'images.csv')
    if data_root is None:
        data_root = json.loads((md / 'config.json').read_text(encoding='utf-8'))['data_root']
    images['path'] = [str(Path(data_root) / r) for r in images['relpath']]
    tasks_file = md / 'tasks.json'
    tasks = json.loads(tasks_file.read_text(encoding='utf-8')) if tasks_file.exists() else {}
    return images, tasks


# ---------------- 데이터 ----------------

def pil_loader(path):
    """torchvision ImageFolder의 기본 로더와 같다."""
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


class ImageList(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = [int(v) for v in labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(pil_loader(self.paths[i])), self.labels[i]


class EvalPreList(Dataset):
    """eval_pre까지 적용한 uint8 텐서 [3, 224, 224]를 돌려준다 (평가 캐시 생성용)."""

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(eval_pre(pil_loader(self.paths[i])))  # HWC uint8
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()


class EvalCache:
    """eval_pre까지 적용해 둔 uint8 이미지. ToTensor·Normalize와 같은 계산을 GPU에서 한다."""

    def __init__(self, work, device):
        path = meta_dir(work) / 'eval224_uint8.pt'
        try:
            self.x = torch.load(path, mmap=True)  # 여러 프로세스가 같은 파일을 공유한다
        except TypeError:  # torch < 2.1
            self.x = torch.load(path)
        self.device = device
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def batches(self, idx, bs=256):
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        for i in range(0, len(idx), bs):
            xb = self.x[idx[i:i + bs]].to(self.device).float().div_(255)
            yield (xb - self.mean) / self.std


# ---------------- 과제와 분할 ----------------

def build_tasks(with_quads=False):
    """6클래스 전체, 재질 5클래스, 재질 클래스의 모든 쌍·삼중 조합 (선택: 4개 조합)."""
    tasks = {'all6': {'classes': CLASSES6, 'pool': 'all'},
             'mat5': {'classes': MATERIALS, 'pool': 'cap'}}
    prefix = {2: 'pair', 3: 'tri', 4: 'quad'}
    for k in ((2, 3, 4) if with_quads else (2, 3)):
        for combo in combinations(MATERIALS, k):
            name = prefix[k] + '_' + '-'.join(ABBR[c] for c in combo)
            tasks[name] = {'classes': list(combo), 'pool': 'cap'}
    return tasks


def task_members(images, task):
    """과제에 속한 이미지의 전역 인덱스와 과제 안 라벨(0..k-1)."""
    classes = task['classes']
    mask = images['cls'].isin(classes)
    if task['pool'] == 'cap':
        mask &= images['cap'].astype(bool)
    sub = images[mask]
    return sub['idx'].to_numpy(), sub['cls'].map({c: i for i, c in enumerate(classes)}).to_numpy()


def task_split(images, task, fold):
    """고정해 둔 fold 번호로 테스트를 정하고, 나머지에서 그룹을 갈라놓지 않고 약 1/5을 검증으로 뗀다."""
    idx, y = task_members(images, task)
    test = images['fold'].to_numpy()[idx] == fold
    tv_idx, tv_y = idx[~test], y[~test]
    groups = images['group'].to_numpy()[tv_idx]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + fold)
    tr, va = next(sgkf.split(tv_idx, tv_y, groups))
    return {'train': (tv_idx[tr], tv_y[tr]), 'val': (tv_idx[va], tv_y[va]), 'test': (idx[test], y[test])}


def team_split(images):
    """compare_fc.ipynb와 같은 층화 70/15/15 분할 (1768/379/380장)."""
    idx, y = images['idx'].to_numpy(), images['label6'].to_numpy()
    train_idx, tmp_idx = train_test_split(idx, test_size=0.3, stratify=y, random_state=TEAM_SPLIT_SEED)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=y[tmp_idx],
                                         random_state=TEAM_SPLIT_SEED)
    return {'train': (train_idx, y[train_idx]), 'val': (val_idx, y[val_idx]), 'test': (test_idx, y[test_idx])}


# ---------------- 모델과 평가 ----------------

def build_model(num_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 노트북과 같은 V1 가중치
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def conv_weights(model):
    return {n: m.weight for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}


def channel_sums(t):
    """conv 가중치 모양 [C_out, C_in, kh, kw]의 텐서를 출력 채널별 합과 입력 채널별 합으로 줄인다."""
    return t.sum(dim=(1, 2, 3)), t.sum(dim=(0, 2, 3))


@torch.no_grad()
def predict_logits(model, cache, idx):
    model.eval()
    return torch.cat([model(xb).float().cpu() for xb in cache.batches(idx)])


def metrics(logits, y):
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    pred = logits.argmax(1)
    return {'acc': (pred == y).float().mean().item(),
            'macro_f1': float(f1_score(y.numpy(), pred.numpy(), average='macro', zero_division=0)),
            'nll': F.cross_entropy(logits, y).item()}


# ---------------- 학습 ----------------

def train_model(model, train_ds, cache, val_idx, val_y, *, seed, total_steps, eval_every, device, workers, tag):
    """compare_fc.ipynb의 학습 루프를 step 기준으로 옮긴 것. 검증 정확도가 가장 높은 시점을 고른다.

    매 step마다 conv 가중치의 movement 점수(-grad * weight, Sanh et al. 2020)를 누적하고,
    평가 시점마다 채널별 합으로 줄여 기록한다. 고른 시점까지의 누적값을 써야 체크포인트와 맞는다.
    """
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=workers, generator=g,
                        pin_memory=device != 'cpu', persistent_workers=workers > 0)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    convs = conv_weights(model)
    pre = {n: w.detach().clone() for n, w in convs.items()}  # 사전학습 conv 가중치
    mov = {n: torch.zeros_like(w) for n, w in convs.items()}  # 누적 -grad * weight
    val_y = torch.as_tensor(np.asarray(val_y), dtype=torch.long)
    hist = {'step': [], 'val_acc': [], 'train_loss': []}
    cum_out, cum_in = {n: [] for n in convs}, {n: [] for n in convs}
    best_acc, best_state, best_i = -1.0, None, -1
    step, loss_sum, n_seen = 0, 0.0, 0
    while step < total_steps:
        for x, y in loader:
            model.train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            with torch.no_grad():
                for n, w in convs.items():
                    mov[n].addcmul_(w.grad, w, value=-1.0)
            opt.step()
            step += 1
            loss_sum += loss.item() * len(y)
            n_seen += len(y)
            if step % eval_every == 0 or step == total_steps:
                logits = predict_logits(model, cache, val_idx)
                acc = (logits.argmax(1) == val_y).float().mean().item()
                hist['step'].append(step)
                hist['val_acc'].append(acc)
                hist['train_loss'].append(loss_sum / n_seen)
                for n in convs:
                    o, i = channel_sums(mov[n])
                    cum_out[n].append(o.cpu())
                    cum_in[n].append(i.cpu())
                if acc > best_acc:  # 노트북처럼 처음으로 최고가 된 시점을 고른다
                    best_acc, best_i = acc, len(hist['step']) - 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                print(f'[{tag}] step {step:4d}  loss={loss_sum / n_seen:.3f}  val_acc={acc:.4f}', flush=True)
                loss_sum, n_seen = 0.0, 0
            if step >= total_steps:
                break
    del loader
    model.load_state_dict(best_state)
    move = {'conv_names': list(convs), 'steps': hist['step'], 'best_i': best_i,
            'cum_out': {n: torch.stack(v) for n, v in cum_out.items()},  # [평가 시점 수, C_out]
            'cum_in': {n: torch.stack(v) for n, v in cum_in.items()},    # [평가 시점 수, C_in]
            'ckpt_out': {}, 'ckpt_in': {}}
    for n, w in convs.items():  # 체크포인트 근사: sum(w_best^2 - w_pre^2)
        o, i = channel_sums(w.detach() ** 2 - pre[n] ** 2)
        move['ckpt_out'][n], move['ckpt_in'][n] = o.cpu(), i.cpu()
    return hist, move


def run_job(images, split, num_classes, cache, out_dir, *, seed, total_steps, eval_every, device, workers,
            tag, extra):
    """기준 모델 하나를 학습하고 out_dir에 저장한다. result.json은 맨 마지막에 써서 완료 표시로 쓴다."""
    t0 = time.time()
    set_seed(seed)
    model = build_model(num_classes)
    (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = split['train'], split['val'], split['test']
    train_ds = ImageList(images['path'].to_numpy()[tr_idx], tr_y, train_tf)
    hist, move = train_model(model, train_ds, cache, va_idx, va_y, seed=seed, total_steps=total_steps,
                             eval_every=eval_every, device=device, workers=workers, tag=tag)
    val_logits = predict_logits(model, cache, va_idx)
    test_logits = predict_logits(model, cache, te_idx)
    info = {**extra, 'seed': seed, 'num_classes': num_classes,
            'n_train': len(tr_idx), 'n_val': len(va_idx), 'n_test': len(te_idx),
            'total_steps': total_steps, 'eval_every': eval_every,
            'best_step': hist['step'][move['best_i']], 'best_val_acc': max(hist['val_acc']),
            'val': metrics(val_logits, va_y), 'test': metrics(test_logits, te_y),
            'val_acc_curve': hist['val_acc'], 'train_loss_curve': hist['train_loss'],
            'time_sec': round(time.time() - t0, 1), 'torch': torch.__version__}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / 'model.pt')
    torch.save(move, out_dir / 'movement.pt')
    np.savez_compressed(out_dir / 'logits.npz', val_idx=va_idx, val_y=va_y, val_logits=val_logits.numpy(),
                        test_idx=te_idx, test_y=te_y, test_logits=test_logits.numpy())
    (out_dir / 'result.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info
