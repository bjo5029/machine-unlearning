# data_es.py: 데이터 로드, 분할, ES 파티션 생성.

import time, numpy as np, torch, hashlib
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, Dataset
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
    """
    (retain_train, forget_train, retain_eval, forget_eval) 로더 생성
    - train_ds  : 증강 O
    - train_eval: 증강 X
    """
    retain_train = Subset(train_ds, retain_idx)
    forget_train = Subset(train_ds, forget_idx)
    retain_eval  = Subset(train_eval_ds, retain_idx)
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
    '집합 ES' 계산: retain 중심(mu_R) 기준
    """
    mu = embs[idx_R].mean(0)
    dF = ((embs[idx_F] - mu)**2).sum(1).mean().item()
    dR = ((embs[idx_R] - mu)**2).sum(1).mean().item()
    return dF - dR

def create_es_partitions_paper(original_model, dataset_for_es, device, batch_size, forget_set_size):
    """
    논문식 ES 파티션:
      - 전역 중심으로부터의 거리 큰 순으로 정렬하여
        Low ES: 가장 먼 구간, Medium ES: 중간, High ES: 가까운 구간
    """
    print("Creating ES partitions (Paper Appendix A.3 method)...")
    t0 = time.time()

    embs = _embed_all(original_model, dataset_for_es, device, batch_size)
    mu = embs.mean(0)
    dists = ((embs - mu)**2).sum(1).numpy()

    order = np.argsort(-dists)  # 먼→가까운
    F = forget_set_size
    N = len(order)
    if 3 * F > N:
        raise ValueError(f"3 * forget_set_size ({3*F}) > dataset size ({N}). Reduce forget_set_size.")

    parts = {
        "Low ES":    order[:F],
        "Medium ES": order[F : 2*F],
        "High ES":   order[2*F : 3*F],
    }

    print(f"ES partitions created in {time.time()-t0:.2f}s")
    return parts

# -------- Optional: balanced variant (class/conf bins) --------
@torch.no_grad()
def create_es_partitions_balanced(original_model, dataset_for_es, device, batch_size, forget_set_size, bins=5):
    """클래스·신뢰도 균형을 고려한 대안 파티션"""
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
    """인덱스 배열을 해시 문자열로 변환(모델 캐싱 등에 사용)"""
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()

def define_forget_set(train_dataset: Dataset, cfg: dict):
    """CONFIG에 따라 Forget Set의 인덱스를 정의합니다."""
    definition = cfg["forget_set_definition"]
    
    if definition == 'random':
        # 'forget_set_size' 대신 'num_to_forget'을 사용하도록 수정
        print(f"Defining forget set: {cfg['num_to_forget']} random samples.") 
        total_size = len(train_dataset)
        forget_indices = np.random.choice(total_size, cfg['num_to_forget'], replace=False)
        return np.sort(forget_indices)

    elif definition == 'class':
        # 'forget_class_id' 대신 'forget_class'를 사용하도록 수정
        class_id = cfg["forget_class"]
        print(f"Defining forget set: all samples from class {class_id}.")
        targets = np.array(train_dataset.targets)
        forget_indices = np.where(targets == class_id)[0]
        return forget_indices
        
    else:
        raise ValueError(f"Unknown forget_set_definition: {definition}")

# ---------- 추가: memorization 점수 로더 ----------
def _load_memorization_scores(npz_path: str, prefer_key: str | None = None) -> np.ndarray:
    """
    npz에서 1D 스코어 벡터를 로드.
    우선순위: prefer_key -> 'memorization' -> 'estimates' -> 'influence'(열 평균 |값|)
    """
    try:
        with np.load(npz_path) as data:
            files = list(data.files)
            key = None
            if prefer_key and prefer_key in data.files:
                key = prefer_key
            elif "memorization" in data.files:
                key = "memorization"
            elif "estimates" in data.files:
                key = "estimates"

            if key is not None:
                s = np.asarray(data[key], dtype=np.float32)
                if s.ndim != 1:
                    raise ValueError(f"{key} must be 1D; got shape {s.shape}")
                return s

            if "influence" in data.files:
                inf = np.asarray(data["influence"], dtype=np.float32)  # [T, N] 예상
                if inf.ndim != 2:
                    raise ValueError(f"'influence' must be 2D; got shape {inf.shape}")
                # 열 기준 평균 |값| → [N]
                s = np.mean(np.abs(inf), axis=0).astype(np.float32)
                return s

            raise KeyError(f"No usable key in NPZ. keys={files}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Memorization NPZ not found: {npz_path}") from e

# ---------- 추가: c-proxy(CE loss) ----------
@torch.no_grad()
def _c_proxy_scores_ce(model, dataset_subset: Dataset, device, batch_size) -> np.ndarray:
    """
    per-sample cross-entropy(y_true) → 값이 낮을수록 '쉬움'으로 간주
    """
    model = model.to(device).eval()
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)
    scores = []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        z = model(x)
        ce = F.cross_entropy(z, y, reduction='none')
        scores.append(ce.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)  # [n_subset]

def partition_forget_set(forget_indices: np.ndarray, train_eval_dataset: Dataset, model, cfg: dict):
    """
    Forget Set을 3 그룹으로 분할.
    반환: [group1, group2, group3]  (각각 '쉬운→어려운' 순서를 유지)
    """
    method = cfg["forget_partitioning_method"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]

    print(f"Partitioning {len(forget_indices)} forget samples using '{method}' method...")

    # 공통: 정렬된 인덱스를 만든 뒤 3등분
    if method == 'es':
        # Forget 부분집합의 중심 기준 거리 계산 → 가까울수록 '쉬움' 가정
        subset_for_es = Subset(train_eval_dataset, forget_indices)
        embs = _embed_all(model, subset_for_es, device, batch_size)
        mu = embs.mean(0)
        dists = ((embs - mu)**2).sum(1).numpy()
        sorted_sub_indices = np.argsort(dists)  # 가까운→먼
        sorted_forget_indices = forget_indices[sorted_sub_indices]

    elif method == 'memorization':
        # memorization 점수 낮을수록 '쉬움' 가정
        npz_path = cfg.get("memorization_score_path", "estimates_results.npz")
        prefer_key = cfg.get("memorization_key", None)  # 예: 'memorization' 또는 'estimates'
        scores_all = _load_memorization_scores(npz_path, prefer_key=prefer_key)  # [N_train] 가정
        if scores_all.shape[0] < np.max(forget_indices)+1:
            raise ValueError(f"Score vector length({scores_all.shape[0]}) < max index({np.max(forget_indices)})")
        forget_scores = scores_all[forget_indices]
        sorted_sub_indices = np.argsort(forget_scores)  # 낮은→높은
        sorted_forget_indices = forget_indices[sorted_sub_indices]

    elif method == 'c_proxy':
        # per-sample CE → 낮을수록 '쉬움'
        subset_for_c = Subset(train_eval_dataset, forget_indices)
        c_scores = _c_proxy_scores_ce(model, subset_for_c, device, batch_size)
        sorted_sub_indices = np.argsort(c_scores)
        sorted_forget_indices = forget_indices[sorted_sub_indices]

    else:
        raise ValueError(f"Unknown partitioning method: {method}")

    # 3분할
    third = len(sorted_forget_indices) // 3
    if third == 0:
        # 샘플이 3 미만인 특수 상황 보호
        partitions = [sorted_forget_indices]
    else:
        partitions = [
            sorted_forget_indices[:third],
            sorted_forget_indices[third: 2*third],
            sorted_forget_indices[2*third:]
        ]
    print(f"Partition sizes: {[len(p) for p in partitions]}")
    return partitions
