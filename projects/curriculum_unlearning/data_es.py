# data_es.py: 데이터 로드, 분할, ES 파티션 생성.

import time, numpy as np, torch, hashlib
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# -------- transforms & loaders (데이터 변환 및 로더) --------
def get_transforms():
    """학습(증강 O), 평가(증강 X)용 데이터 변환"""
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    return tf_train, tf_eval

def load_cifar10_with_train_eval(data_root):
    """CIFAR-10 데이터셋을 (train, train_eval, test)로 불러옴"""
    tf_train, tf_eval = get_transforms()
    train      = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf_train)
    train_eval = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf_eval)
    test       = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_eval)
    return train, train_eval, test

def split_retain_forget(indices_len, forget_indices):
    """전체 인덱스에서 forget 인덱스를 제외하여 retain 인덱스를 만듦"""
    all_idx = np.arange(indices_len)
    retain_idx = np.setdiff1d(all_idx, forget_indices, assume_unique=True)
    return retain_idx, forget_indices

def build_loaders(train_ds, train_eval_ds, retain_idx, forget_idx, batch, shuffle_train=True):
    """주어진 인덱스를 바탕으로 4종류의 데이터 로더 생성
    (retain_train, forget_train, retain_eval, forget_eval)
    """
    retain_train = Subset(train_ds, retain_idx)   # augment O
    forget_train = Subset(train_ds, forget_idx)
    retain_eval  = Subset(train_eval_ds, retain_idx)  # augment X
    forget_eval  = Subset(train_eval_ds, forget_idx)

    retain_train_loader = DataLoader(retain_train, batch_size=batch, shuffle=shuffle_train)
    forget_train_loader = DataLoader(forget_train, batch_size=batch, shuffle=shuffle_train)
    retain_eval_loader  = DataLoader(retain_eval,  batch_size=batch, shuffle=False)
    forget_eval_loader  = DataLoader(forget_eval,  batch_size=batch, shuffle=False)
    return retain_train_loader, forget_train_loader, retain_eval_loader, forget_eval_loader

# -------- ES (paper-like) --------
@torch.no_grad()
def _embed_all(model, dataset, device, batch_size):
    """모델의 특징 추출기(마지막 레이어 제외)를 사용해 모든 데이터의 임베딩(특징 벡터)을 추출"""
    model = model.to(device).eval()
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs = []
    for x, _ in loader:
        x = x.to(device)
        e = extractor(x).squeeze()
        embs.append(e.detach().cpu())
    return torch.cat(embs, 0)  # [N, D]

def _set_es(embs, idx_R, idx_F):
    """
    '집합 ES'를 계산
    Retain 데이터의 중심을 기준으로 Forget과 Retain 데이터가 각각 얼마나 떨어져있는지 그 차이를 계산
    """
    # 전역 센트로이드 기반 간단 set-ES: forget의 평균 제곱거리 - retain의 평균 제곱거리
    mu = embs[idx_R].mean(0) # Retain 데이터의 중심점(센트로이드)
    dF = ((embs[idx_F] - mu)**2).sum(1).mean().item() # Forget 데이터들이 Retain 중심에서 떨어진 평균 거리
    dR = ((embs[idx_R] - mu)**2).sum(1).mean().item() # Retain 데이터들이 자기 중심에서 떨어진 평균 거리 (분산)
    return dF - dR # 이 값이 클수록 Forget 데이터가 Retain 분포에서 더 이질적임을 의미

def create_es_partitions_paper(original_model, dataset_for_es, device, batch_size, forget_set_size):
    """논문 방식과 유사하게 ES 파티션을 생성"""
    print("Creating ES partitions (paper-accurate)...")
    embs = _embed_all(original_model, dataset_for_es, device, batch_size)  # 모든 데이터의 임베딩 추출
    mu = embs.mean(0) # 전체 데이터의 중심점
    dists = ((embs - mu)**2).sum(1).numpy() # 각 데이터와 전체 중심점 간의 거리 계산

    order = np.argsort(dists)  # 거리가 가까운 순서대로 데이터 인덱스 정렬
    N = len(order); F = forget_set_size
    if 3*F > N:
        raise ValueError(f"3*forget_set_size ({3*F}) > N ({N}). Reduce forget_set_size.")

    # 3개의 후보 블록(가까운 그룹, 중간 그룹, 먼 그룹) 생성
    cand1 = order[:F]
    cand2 = order[F:2*F]
    cand3 = order[2*F:3*F]

    all_idx = np.arange(N)
    def block_es(cand):
        idx_F = cand
        idx_R = np.setdiff1d(all_idx, idx_F, assume_unique=False)
        return _set_es(embs, idx_R, idx_F)

    # 각 블록의 '집합 ES' 점수 계산
    es1, es2, es3 = block_es(cand1), block_es(cand2), block_es(cand3)
    blocks = [("B1", cand1, es1), ("B2", cand2, es2), ("B3", cand3, es3)]

    # ES 점수가 낮은 순서대로 정렬하여 Low, Medium, High 라벨 부여
    blocks_sorted = sorted(blocks, key=lambda t: t[2])  # ascending
    parts = {"Low ES": np.array(blocks_sorted[0][1]),
             "Medium ES": np.array(blocks_sorted[1][1]),
             "High ES": np.array(blocks_sorted[2][1])}
    print(f"ES scores (ascending): {[round(b[2],6) for b in blocks_sorted]}")
    return parts

# -------- Optional: balanced variant (class/conf bins) --------
@torch.no_grad()
def create_es_partitions_balanced(original_model, dataset_for_es, device, batch_size, forget_set_size, bins=5):
    """클래스별로, 그리고 신뢰도 구간별로 균등하게 샘플을 뽑아 더 대표성 있는 Forget Set을 만드는 대안적인 파티션 방식"""
    model = original_model.to(device).eval()
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    loader = DataLoader(dataset_for_es, batch_size=batch_size, shuffle=False)

    all_idx, all_y, all_conf, embs_list = [], [], [], []
    seen = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        z = model(x); p = F.softmax(z, 1)
        conf = p[torch.arange(x.size(0)), y]
        emb = extractor(x).squeeze()
        bs = x.size(0)
        all_idx.extend(range(seen, seen+bs))
        all_y.extend(y.tolist()); all_conf.extend(conf.detach().cpu().tolist())
        embs_list.append(emb.detach().cpu()); seen += bs

    embs = torch.cat(embs_list, 0)
    mu = embs.mean(0); dists = ((embs - mu)**2).sum(1).numpy()
    all_idx = np.asarray(all_idx); all_y = np.asarray(all_y); all_conf = np.asarray(all_conf)

    k, fs = int(all_y.max())+1, forget_set_size
    fs_per_class = fs // k; remainder = fs - fs_per_class*k
    rng = np.random.default_rng(123)
    forget_indices = []
    for c in range(k):
        need_c = fs_per_class + (1 if c < remainder else 0)
        m = (all_y == c); idx_c = all_idx[m]; conf_c = all_conf[m]; d_c = dists[m]
        if idx_c.size == 0 or need_c == 0: continue
        qs = np.quantile(conf_c, np.linspace(0,1,bins+1))
        chosen = []
        per_bin = max(1, need_c // bins)
        for b in range(bins):
            lo, hi = qs[b], qs[b+1]
            mm = (conf_c >= lo) & (conf_c <= hi) if b == bins-1 else (conf_c >= lo) & (conf_c < hi)
            cand = idx_c[mm]
            if cand.size == 0: continue
            take = min(per_bin, cand.size)
            sel = rng.choice(cand, size=take, replace=False)
            chosen.extend(sel.tolist())
        if len(chosen) < need_c:
            remain = need_c - len(chosen)
            not_mask = ~np.isin(idx_c, np.asarray(chosen))
            cand = idx_c[not_mask]; cand_d = d_c[not_mask]
            if cand.size > 0:
                extra = cand[np.argsort(-cand_d)[:min(remain, cand.size)]]
                chosen.extend(extra.tolist())
        forget_indices.extend(chosen[:need_c])

    forget_indices = np.asarray(forget_indices)
    d_forget = dists[forget_indices]
    order = np.argsort(d_forget)  # 작은→큰
    fs = forget_indices[order]; third = len(fs)//3
    return {"Low ES": fs[:third], "Medium ES": fs[third:2*third], "High ES": fs[2*third:3*third]}

def hash_indices(indices: np.ndarray) -> str:
    """
    인덱스 배열을 고유한 해시값(문자열)으로 변환함
    모델 캐싱에 사용
    """
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()
