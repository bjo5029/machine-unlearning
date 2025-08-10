# data_es.py: CIFAR-10 로드, retain/forget 분할, ES 파티션 생성.

import time, numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn as nn
import torch, numpy as np
import torch.nn.functional as F

def get_transforms():
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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
    indices_len: 전체 길이(len(train_*))
    반환: retain_idx, forget_idx (numpy arrays)
    """
    all_idx = np.arange(indices_len)
    retain_idx = np.setdiff1d(all_idx, forget_indices, assume_unique=True)
    return retain_idx, forget_indices

def create_es_partitions(original_model, dataset_for_es, device, batch_size, forget_set_size):
    """
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
    클래스 균형 + 난이도(=confidence) 균형을 맞춰서 ES 파티션 생성.
    - 평가 변환(augment X) 데이터셋에서 임베딩과 confidence를 측정
    - 클래스별 동일 수(fs_per_class) 만큼 선별
    - 각 클래스 내에서 confidence를 bins개 구간으로 나눠 균등 샘플링
    - 각 bin 안에서는 ES(centroid 거리) 큰 순으로 뽑거나, 랜덤 샘플링 선택 가능
    """
    import torch, numpy as np
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import torch.nn as nn

    model = original_model.to(device).eval()
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    loader = DataLoader(dataset_for_es, batch_size=batch_size, shuffle=False)
    all_idx, all_y, all_conf, all_es = [], [], [], []

    # 1) 임베딩/ confidence 계산
    with torch.no_grad():
        embs_list = []
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            logits = model(x)
            probs  = F.softmax(logits, 1)
            conf   = probs[torch.arange(x.size(0)), y.to(device)]
            emb    = extractor(x).squeeze()

            # 수집
            start = i * batch_size
            for j in range(x.size(0)):
                all_idx.append(start + j)
                all_y.append(int(y[j]))
                all_conf.append(float(conf[j].cpu()))
            embs_list.append(emb.detach().cpu())

        embs = torch.cat(embs_list, 0)
        centroid = embs.mean(0)
        dists = ((embs - centroid) ** 2).sum(1).numpy()  # ES로 쓸 값

    all_idx = np.array(all_idx)
    all_y   = np.array(all_y)
    all_conf= np.array(all_conf)
    all_es  = np.array(dists)

    # 2) 클래스별/난이도별 균형 샘플링
    k = 10
    fs = forget_set_size
    fs_per_class = fs // k
    remainder = fs - fs_per_class * k  # 남는 건 앞쪽 클래스에 1씩

    forget_indices = []

    for c in range(k):
        need_c = fs_per_class + (1 if c < remainder else 0)
        mask_c = (all_y == c)
        idx_c  = all_idx[mask_c]
        conf_c = all_conf[mask_c]
        es_c   = all_es[mask_c]

        if len(idx_c) == 0:
            continue

        # confidence로 bins개 구간 나누기
        qs = np.quantile(conf_c, np.linspace(0, 1, bins+1))
        chosen_c = []

        # 각 bin에서 동일 수로 뽑기
        per_bin = max(1, need_c // bins)
        leftover = need_c - per_bin * bins

        # bin별 선택: ES 큰 순으로 뽑거나(‘어렵게’/‘특이한’), 랜덤 뽑기(편향 최소화)
        # 여기선 **랜덤**으로 뽑고, 남는 leftover는 ES 큰 순으로 보강
        for b in range(bins):
            lo, hi = qs[b], qs[b+1]
            m = (conf_c >= lo) & (conf_c <= hi) if b == bins-1 else (conf_c >= lo) & (conf_c < hi)

            cand_idx = idx_c[m]
            cand_es  = es_c[m]

            if len(cand_idx) == 0:
                continue

            # 2a) 우선 랜덤 추출
            take = min(per_bin, len(cand_idx))
            sel = np.random.choice(cand_idx, size=take, replace=False)
            chosen_c.extend(sel.tolist())

        # 2b) 남은 개수는 ES 큰 순으로 보강(난이도/특이도 섞기)
        if len(chosen_c) < need_c:
            remain = need_c - len(chosen_c)
            # 아직 안 뽑힌 후보들
            mask_not = np.isin(idx_c, np.array(chosen_c), invert=True)
            cand_idx = idx_c[mask_not]
            cand_es  = es_c[mask_not]
            if len(cand_idx) > 0:
                order = np.argsort(-cand_es)  # ES 큰 순
                sel = cand_idx[order[:min(remain, len(cand_idx))]]
                chosen_c.extend(sel.tolist())

        forget_indices.extend(chosen_c[:need_c])

    forget_indices = np.array(forget_indices)

    # 3) 파티션(세 구간) 나누기: 이제는 "난이도"로 다시 쪼개기보다,
    #    균형 구성된 forget_indices를 세 등분(혹은 원하는 대로)만 해도 됨.
    #    여기선 기존 테이블 호환 위해 ES 기준 오름차순으로 정렬 후 세 파트.
    es_forget = all_es[forget_indices]
    order = np.argsort(es_forget)  # 가까운→먼 (원하는 정의로 변경 가능)
    forget_sorted = forget_indices[order]

    # 3등분
    third = len(forget_sorted) // 3
    parts = {
        "Low ES":    forget_sorted[:third],
        "Medium ES": forget_sorted[third:2*third],
        "High ES":   forget_sorted[2*third:3*third],
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
