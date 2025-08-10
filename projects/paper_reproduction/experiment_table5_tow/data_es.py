import time, numpy as np, torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Subset

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

def load_cifar10(data_root="./data"):
    tf_train, tf_test = get_transforms()
    train = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf_train)
    test  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_test)
    full_train_eval = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_test)
    return train, test, full_train_eval

def split_retain_forget(dataset, forget_indices):
    all_idx = np.arange(len(dataset))
    retain_idx = np.setdiff1d(all_idx, forget_indices, assume_unique=True)
    return Subset(dataset, retain_idx), Subset(dataset, forget_indices)

def create_es_partitions(original_model, train_dataset, device, batch_size, forget_set_size):
    """
    ES 파티션: penultimate 임베딩 -> 중심거리 내림차순 정렬 -> Low/Medium/High 각 fs개
    """
    print("Creating ES partitions...")
    t0 = time.time()
    extractor = nn.Sequential(*list(original_model.children())[:-1]).to(device).eval()

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    embs = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if (i+1) % 40 == 0:
                print(f"  Batch {i+1}/{len(loader)}")
            e = extractor(x.to(device)).squeeze().cpu()
            embs.append(e)
    embs = torch.cat(embs, 0)

    centroid = embs.mean(0)
    dists = ((embs - centroid) ** 2).sum(1)
    idx = dists.argsort(descending=True).numpy()

    fs = forget_set_size
    parts = {
        "Low ES": idx[:fs],
        "Medium ES": idx[fs:2*fs],
        "High ES": idx[2*fs:3*fs],
    }
    print(f"ES partitions created in {time.time()-t0:.2f}s.")
    return parts
