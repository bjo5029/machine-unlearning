import time, numpy as np, torch, hashlib
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

def get_transforms():
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
    tf_train, tf_eval = get_transforms()
    train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    train_eval = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_eval)
    test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_eval)
    return train, train_eval, test

def split_retain_forget(indices_len, forget_indices):
    all_idx = np.arange(indices_len)
    retain_idx = np.setdiff1d(all_idx, forget_indices, assume_unique=True)
    return retain_idx, forget_indices

def build_loaders(train_ds, train_eval_ds, retain_idx, forget_idx, batch, shuffle_train=True):
    retain_train = Subset(train_ds, retain_idx) if len(retain_idx) > 0 else []
    forget_train = Subset(train_ds, forget_idx) if len(forget_idx) > 0 else []
    retain_eval  = Subset(train_eval_ds, retain_idx) if len(retain_idx) > 0 else []
    forget_eval  = Subset(train_eval_ds, forget_idx) if len(forget_idx) > 0 else []

    retain_train_loader = DataLoader(retain_train, batch_size=batch, shuffle=shuffle_train) if retain_train else None
    forget_train_loader = DataLoader(forget_train, batch_size=batch, shuffle=shuffle_train) if forget_train else None
    retain_eval_loader  = DataLoader(retain_eval,  batch_size=batch, shuffle=False) if retain_eval else None
    forget_eval_loader  = DataLoader(forget_eval,  batch_size=batch, shuffle=False) if forget_eval else None
    return retain_train_loader, forget_train_loader, retain_eval_loader, forget_eval_loader

@torch.no_grad()
def _embed_all(model, dataset, device, batch_size):
    model = model.to(device).eval()
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs = []
    for x, _ in loader:
        x = x.to(device)
        e = extractor(x).squeeze()
        embs.append(e.detach().cpu())
    return torch.cat(embs, 0)

def _load_memorization_scores(npz_path: str, prefer_key: str | None = None) -> np.ndarray:
    try:
        with np.load(npz_path) as data:
            files = list(data.files)
            key = None
            if prefer_key and prefer_key in data.files: key = prefer_key
            elif "memorization" in data.files: key = "memorization"
            elif "estimates" in data.files: key = "estimates"
            if key is not None:
                s = np.asarray(data[key], dtype=np.float32)
                if s.ndim != 1: raise ValueError(f"{key} must be 1D; got shape {s.shape}")
                return s
            if "influence" in data.files:
                inf = np.asarray(data["influence"], dtype=np.float32)
                if inf.ndim != 2: raise ValueError(f"'influence' must be 2D; got shape {inf.shape}")
                s = np.mean(np.abs(inf), axis=0).astype(np.float32)
                return s
            raise KeyError(f"No usable key in NPZ. keys={files}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Memorization NPZ not found: {npz_path}") from e

@torch.no_grad()
def _c_proxy_scores_ce(model, dataset_subset: Dataset, device, batch_size) -> np.ndarray:
    model = model.to(device).eval()
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)
    scores = []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        z = model(x)
        ce = F.cross_entropy(z, y, reduction='none')
        scores.append(ce.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)

def hash_indices(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()

def define_forget_set(train_dataset: Dataset, cfg: dict):
    definition = cfg["forget_set_definition"]
    if definition == 'random':
        print(f"Defining forget set: {cfg['num_to_forget']} random samples.")
        total_size = len(train_dataset)
        forget_indices = np.random.choice(total_size, cfg['num_to_forget'], replace=False)
        return np.sort(forget_indices)
    elif definition == 'class':
        class_id = cfg["forget_class"]
        print(f"Defining forget set: all samples from class {class_id}.")
        targets = np.array(train_dataset.targets)
        forget_indices = np.where(targets == class_id)[0]
        return forget_indices
    else:
        raise ValueError(f"Unknown forget_set_definition: {definition}")

def _get_sorted_indices(indices: np.ndarray, dataset: Dataset, model, cfg: dict, method: str, global_mu=None):
    """주어진 인덱스에 대해 특정 방법(method)으로 점수를 매기고 정렬된 인덱스를 반환"""
    device = cfg["device"]; batch_size = cfg["batch_size"]
    print(f"  - Scoring {len(indices)} samples using '{method}'...")
    if method == 'random':
        np.random.shuffle(indices)
        return indices
    subset = Subset(dataset, indices)
    if method == 'es':
        if global_mu is None: raise ValueError("global_mu is required for 'es' method.")
        embs = _embed_all(model, subset, device, batch_size)
        scores = ((embs - global_mu.to(embs.device))**2).sum(1).numpy()
    elif method == 'memorization':
        npz_path = cfg.get("memorization_score_path", "estimates_results.npz")
        scores_all = _load_memorization_scores(npz_path)
        if len(scores_all) < np.max(indices) + 1: raise ValueError("Score vector length doesn't match.")
        scores = scores_all[indices]
    elif method == 'c_proxy':
        scores = _c_proxy_scores_ce(model, subset, device, batch_size)
    else:
        raise ValueError(f"Unknown partitioning method: {method}")
    return indices[np.argsort(scores)]

def _partition_indices(sorted_indices: np.ndarray, ordering: str, granularity: str):
    """미리 정렬된 인덱스를 받아 파티션으로 분할"""
    if ordering == "hard_first":
        sorted_indices = sorted_indices[::-1]
    if granularity == 'sample':
        partitions = [sorted_indices] if len(sorted_indices) > 0 else []
    else:
        third = len(sorted_indices) // 3
        if third == 0:
            partitions = [sorted_indices] if len(sorted_indices) > 0 else []
        else:
            partitions = [sorted_indices[:third], sorted_indices[third: 2*third], sorted_indices[2*third:]]
    print(f"    Partition sizes: {[len(p) for p in partitions]}")
    return partitions
