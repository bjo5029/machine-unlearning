# data_es.py: CIFAR-10 로드, retain/forget 분할, ES 파티션 생성.

import time, numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn as nn
import torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_transforms():
    """데이터 증강 및 정규화 정의"""
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 이미지를 36x36으로 패딩한 후 무작위로 32x32 크기로 자름 (위치 변화)
        transforms.RandomHorizontalFlip(), # 50% 확률로 이미지 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    return tf_train, tf_test

def load_cifar10_with_train_eval(data_root):
    """
    train(augment O), train_eval(augment X), test(augment X) 반환
    """
    tf_train, tf_test = get_transforms()
    train      = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf_train)
    train_eval = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf_test)
    test       = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_test)
    return train, train_eval, test

def split_retain_forget(indices_len, forget_indices):
    """
    전체 인덱스에서 forget 인덱스를 제외하여 retain 인덱스를 만듬
    indices_len: 전체 길이(len(train_*))
    반환: retain_idx, forget_idx (numpy arrays)
    """
    all_idx = np.arange(indices_len)
    retain_idx = np.setdiff1d(all_idx, forget_indices, assume_unique=True)
    return retain_idx, forget_indices

# --- ES 파티션 생성 함수들 ---

def create_es_partitions(original_model, dataset_for_es, device, batch_size, forget_set_size):
    """
    전역 센트로이드로부터의 거리가 먼 순서대로 High/Medium/Low ES를 할당하는 가장 간단한 방식

    ES 파티션은 평가 안정성을 위해 기본적으로 'augment X' 데이터셋에서 생성 권장.
    dataset_for_es: 보통 train_eval을 넣어라.
    """
    print("Creating ES partitions (eval transforms)...")
    t0 = time.time()
    extractor = nn.Sequential(*list(original_model.children())[:-1]).to(device).eval()
    loader = torch.utils.data.DataLoader(dataset_for_es, batch_size=batch_size, shuffle=False)
    embs = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if (i+1) % 40 == 0: print(f"  batch {i+1}/{len(loader)}")
            e = extractor(x.to(device)).squeeze().cpu()
            embs.append(e)
    embs = torch.cat(embs, 0)
    centroid = embs.mean(0)
    dists = ((embs - centroid) ** 2).sum(1)
    idx = dists.argsort(descending=True).numpy()  # 거리 큰 순
    fs = forget_set_size
    parts = {"Low ES": idx[:fs], "Medium ES": idx[fs:2*fs], "High ES": idx[2*fs:3*fs]}
    print(f"ES partitions in {time.time()-t0:.2f}s")
    return parts

def create_es_partitions_balanced(original_model, dataset_for_es, device, batch_size, forget_set_size, bins=5):
    """
    클래스별로 균등하게 샘플을 뽑되, 각 클래스 내에서 '클래스 상대적 ES'와 '신뢰도(confidence)'를 고려하여 균형 잡힌 forget set을 만드는 복잡한 방식

    클래스/난이도(confidence) 균형 유지 + '클래스-상대 ES'로 파티션 생성.
    ES(x) = d_self(x) - min_{c≠y} d_other(x)  (값이 클수록 harder/entangled)
    """
    model = original_model.to(device).eval()
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    loader = DataLoader(dataset_for_es, batch_size=batch_size, shuffle=False)
    all_idx, all_y, all_conf = [], [], []
    embs_list = []

    # 1) 임베딩 + confidence 수집
    with torch.no_grad():
        seen = 0
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            p = F.softmax(z, 1)
            conf = p[torch.arange(x.size(0)), y.to(device)]
            emb = extractor(x).squeeze()      # [B, D] (resnet18 avgpool 출력)

            bs = x.size(0)
            all_idx.extend(range(seen, seen + bs))
            all_y.extend(y.tolist())
            all_conf.extend(conf.detach().cpu().tolist())
            embs_list.append(emb.detach().cpu())
            seen += bs

    embs = torch.cat(embs_list, 0)           # [N, D]  (cpu tensor)
    all_idx = np.asarray(all_idx)
    all_y   = np.asarray(all_y)
    all_conf= np.asarray(all_conf)

    # 2) 클래스 중심 계산
    k = int(all_y.max()) + 1
    y_t = torch.tensor(all_y, dtype=torch.long)
    mu_list = []
    for c in range(k):
        mask = (y_t == c)
        # (희박 케이스 방어) 해당 클래스 샘플이 없다면 전체 평균으로 대체
        mu_c = embs[mask].mean(0) if mask.any() else embs.mean(0)
        mu_list.append(mu_c)
    mu = torch.stack(mu_list, 0)             # [C, D]

    # 3) 클래스-상대 ES 계산
    #    d_self: 자기 클래스까지의 제곱거리
    #    d_other: 타 클래스 중심까지의 최소 제곱거리
    dist2 = ((embs[:, None, :] - mu[None, :, :])**2).sum(-1)  # [N, C]
    idx = torch.arange(embs.size(0))
    d_self = dist2[idx, y_t]                                  # [N]

    dist2_masked = dist2.clone()
    dist2_masked[idx, y_t] = float('inf')
    d_other = dist2_masked.min(1).values                      # [N]

    es_score = (d_self - d_other).cpu().numpy()               # 값↑ = harder/entangled

    # 4) 클래스/난이도(confidence) 균형 샘플링
    fs = forget_set_size
    fs_per_class = fs // k
    remainder = fs - fs_per_class * k

    forget_indices = []
    rng = np.random.default_rng()  # 재현성 원하면 밖에서 np.random.seed(...) 호출

    for c in range(k):
        need_c = fs_per_class + (1 if c < remainder else 0)
        mask_c = (all_y == c)
        idx_c  = all_idx[mask_c]
        conf_c = all_conf[mask_c]
        es_c   = es_score[mask_c]

        if idx_c.size == 0 or need_c == 0:
            continue

        # confidence를 bins 구간으로 등분
        qs = np.quantile(conf_c, np.linspace(0, 1, bins+1))
        chosen_c = []

        per_bin = max(1, need_c // bins)

        # 각 bin에서 우선 랜덤 추출
        for b in range(bins):
            lo, hi = qs[b], qs[b+1]
            m = (conf_c >= lo) & (conf_c <= hi) if b == bins-1 else (conf_c >= lo) & (conf_c < hi)
            cand_idx = idx_c[m]
            if cand_idx.size == 0:
                continue
            take = min(per_bin, cand_idx.size)
            sel = rng.choice(cand_idx, size=take, replace=False)
            chosen_c.extend(sel.tolist())

        # 남은 몫은 ES 큰 순(=더 hard)으로 채우기
        if len(chosen_c) < need_c:
            remain = need_c - len(chosen_c)
            not_mask = ~np.isin(idx_c, np.asarray(chosen_c))
            cand_idx = idx_c[not_mask]
            cand_es  = es_c[not_mask]
            if cand_idx.size > 0:
                order = np.argsort(-cand_es)  # ES 큰 순
                extra = cand_idx[order[:min(remain, cand_idx.size)]]
                chosen_c.extend(extra.tolist())

        forget_indices.extend(chosen_c[:need_c])

    forget_indices = np.asarray(forget_indices)

    # 5) High/Medium/Low 파티션 (ES 내림차순 정렬)
    es_forget = es_score[forget_indices]
    order = np.argsort(-es_forget)          # 큰 ES = harder → High ES 앞쪽
    sorted_idx = forget_indices[order]

    third = len(sorted_idx) // 3
    parts = {
        "High ES":   sorted_idx[:third],
        "Medium ES": sorted_idx[third:2*third],
        "Low ES":    sorted_idx[2*third:3*third],
    }
    return parts

def _compute_per_sample_loss_and_labels(model, dataset_eval, device, batch_size=256):
    loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    model.eval(); losses=[]; labels=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device)
            z = model(x)
            l = F.cross_entropy(z, y, reduction='none').detach().cpu().numpy()
            losses.append(l); labels.append(y.cpu().numpy())
    return np.concatenate(losses), np.concatenate(labels)

def create_loss_partitions(model, dataset_eval, device, forget_set_size, num_classes=10, batch_size=256):
    """
    모델의 예측 손실(loss)이 큰 (어려운) 샘플 순으로 High/Medium/Low 파티션을 구성하는 방식
    
    per-sample CE loss로 각 클래스별 상위 구간을 High/Medium/Low로 쪼개서
    기존 키값("Low ES","Medium ES","High ES")을 유지한 채 dict 반환
    """
    losses, labels = _compute_per_sample_loss_and_labels(model, dataset_eval, device, batch_size)
    per_class_take = forget_set_size // num_classes

    idx_by_cls = []
    for c in range(num_classes):
        idx_c = np.where(labels==c)[0]
        # 어려운(=loss 큰) 순서대로
        idx_sorted = idx_c[np.argsort(-losses[idx_c])]
        idx_by_cls.append(idx_sorted)

    high_idx, med_idx, low_idx = [], [], []
    for c in range(num_classes):
        s = idx_by_cls[c]
        high_idx += list(s[:per_class_take])
        med_idx  += list(s[per_class_take:2*per_class_take])
        low_idx  += list(s[2*per_class_take:3*per_class_take])

    return {
        "Low ES":    np.array(low_idx),   # 비교적 쉬운 쪽
        "Medium ES": np.array(med_idx),
        "High ES":   np.array(high_idx),  # 가장 어려운 쪽
    }

@torch.no_grad()
def _embed_all(original_model, dataset_eval, device, batch_size):
    """모델의 특징 추출기를 사용해 모든 데이터의 임베딩(특징 벡터)을 추출합니다."""
    loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)
    model = original_model.to(device).eval()
    # 마지막 FC 레이어를 제외한 특징 추출기
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    embs = []
    for x, _ in loader:
        x = x.to(device)
        e = extractor(x).squeeze()  # [B, D]
        embs.append(e.detach().cpu())
    return torch.cat(embs, 0)  # [N, D] (cpu)

def create_es_partitions_paper(original_model, dataset_for_es, device, batch_size, forget_set_size):
    """
    논문 부록 A.3의 절차 재현:
      1) 전체 데이터의 중심점(global centroid)으로부터 각 데이터까지의 거리 계산
      2) 거리가 먼 순서대로 데이터 정렬
      3) 정렬된 순서에 따라 직접 Low, Medium, High ES 
         - Low ES: 가장 먼(highest distance) 데이터 그룹
         - High ES: 점차 가까워지는(progressively lower distance) 데이터 그룹
    """
    print("Creating ES partitions (Paper Appendix A.3 method)...")
    t0 = time.time()

    # 1) 모든 데이터의 임베딩 및 중심점으로부터의 거리 계산
    embs = _embed_all(original_model, dataset_for_es, device, batch_size)
    mu = embs.mean(0)
    dists = ((embs - mu)**2).sum(1).numpy()

    # 2) 거리가 '먼' 순서대로 인덱스를 정렬 (내림차순)
    order = np.argsort(-dists)

    # 3) 정렬된 순서에 따라 직접 그룹 명명
    F = forget_set_size
    if 3 * F > len(order):
        raise ValueError(f"3 * forget_set_size ({3*F}) is larger than the dataset size ({len(order)}). Please reduce forget_set_size.")

    parts = {
        "Low ES":    order[:F],          # 가장 먼 3000개
        "Medium ES": order[F : 2*F],     # 그 다음 3000개
        "High ES":   order[2*F : 3*F],   # 그 다음 3000개 (가장 가까운 그룹에 속함)
    }

    print(f"ES partitions created in {time.time()-t0:.2f}s")
    # ES 점수를 직접 계산하지 않으므로, 관련 로그는 제거합니다.
    return parts

def _set_es(embs: torch.Tensor, idx_R: np.ndarray, idx_F: np.ndarray, eps=1e-12) -> float:
    """
    집합 ES = (within_R + within_F) / (0.5 * (||mu_R - mu||^2 + ||mu_F - mu||^2))
    within은 평균제곱거리, mu는 전체평균.
    """
    z = embs  # [N, D] cpu tensor
    mu  = z.mean(0)
    R   = z[idx_R]; F = z[idx_F]
    muR = R.mean(0); muF = F.mean(0)

    within_R = ((R - muR)**2).sum(1).mean().item()
    within_F = ((F - muF)**2).sum(1).mean().item()
    between  = 0.5 * (((muR - mu)**2).sum().item() + ((muF - mu)**2).sum().item())
    return (within_R + within_F) / (between + eps)
