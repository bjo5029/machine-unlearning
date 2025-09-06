import copy, itertools, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------- Random-label / SalUn용 데이터셋 --------
class RelabelDataset(Dataset):
    def __init__(self, ds, num_classes=10):
        self.ds = ds; self.k = num_classes
        self.new_labels = [torch.randint(0, self.k, (1,)).item() for _ in range(len(ds))]
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, y = self.ds[idx]
        nl = self.new_labels[idx]
        while nl == y:
            nl = torch.randint(0, self.k, (1,)).item()
        return img, nl

# -------- 공통 미니 학습 루프 --------
def _train_simple(model, loader, epochs, lr, device, momentum, weight_decay):
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()

# -------- 각 언러닝 기법 --------
def unlearn_finetune(original_model, retain_loader, cfg):
    m = copy.deepcopy(original_model)
    _train_simple(m, retain_loader, cfg["unlearn_epochs"], cfg["unlearn_lr"],
                  cfg["device"], cfg["momentum"], cfg["weight_decay"])
    return m

def unlearn_neggrad(original_model, forget_loader, cfg):
    m = copy.deepcopy(original_model); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr_neggrad"])
    m.train()
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in forget_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad(); loss = -crit(m(x), y); loss.backward(); opt.step()
    return m

def unlearn_l1_sparse(original_model, retain_loader, cfg):
    m = copy.deepcopy(original_model); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"], momentum=cfg["momentum"])
    m.train()
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in retain_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad()
            l1 = sum(p.abs().sum() for p in m.parameters())
            loss = crit(m(x), y) + cfg["l1_lambda"] * l1
            loss.backward(); opt.step()
    return m

def unlearn_neggrad_plus(original_model, retain_loader, forget_loader, cfg):
    m = copy.deepcopy(original_model); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"])
    m.train(); r_iter = itertools.cycle(retain_loader)
    for _ in range(cfg["unlearn_epochs"]):
        for fx, fy in forget_loader:
            rx, ry = next(r_iter)
            rx, ry = rx.to(cfg["device"]), ry.to(cfg["device"])
            fx, fy = fx.to(cfg["device"]), fy.to(cfg["device"])
            opt.zero_grad()
            loss = crit(m(rx), ry) - cfg["neggrad_plus_alpha"] * crit(m(fx), fy)
            loss.backward(); opt.step()
    return m

def unlearn_random_label(original_model, forget_set, cfg):
    m = copy.deepcopy(original_model)
    loader = DataLoader(RelabelDataset(forget_set), batch_size=cfg["batch_size"], shuffle=True)
    _train_simple(m, loader, cfg["unlearn_epochs"], cfg["unlearn_lr"],
                  cfg["device"], cfg["momentum"], cfg["weight_decay"])
    return m

def unlearn_salun(original_model, forget_set, cfg):
    m = copy.deepcopy(original_model); crit = nn.CrossEntropyLoss()
    # 1) saliency 적산
    sal = [torch.zeros_like(p) for p in m.parameters()]
    f_loader = DataLoader(forget_set, batch_size=cfg["batch_size"])
    m.train()
    for x, y in f_loader:
        x, y = x.to(cfg["device"]), y.to(cfg["device"])
        m.zero_grad(); loss = crit(m(x), y); loss.backward()
        for i, p in enumerate(m.parameters()):
            if p.grad is not None:
                sal[i] += p.grad.abs()
    # 2) 마스크
    flat = torch.cat([s.flatten() for s in sal])
    k = max(1, int(len(flat) * cfg["salun_sparsity"]))
    th, _ = torch.kthvalue(flat, k)
    masks = [(s > th).float() for s in sal]
    # 3) 랜덤 라벨로 salient 파라미터만 업데이트
    rl = DataLoader(RelabelDataset(forget_set), batch_size=cfg["batch_size"], shuffle=True)
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"], momentum=cfg["momentum"])
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in rl:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad(); loss = crit(m(x), y); loss.backward()
            for i, p in enumerate(m.parameters()):
                if p.grad is not None:
                    p.grad *= masks[i]
            opt.step()
    return m
