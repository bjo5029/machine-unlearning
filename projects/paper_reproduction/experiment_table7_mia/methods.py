# methods.py: Fine-tune, L1-sparse, NegGrad, NegGrad+, SCRUB, SalUn, Random-label 및 fixed 변형.

import copy, torch, itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class RelabelDataset(Dataset):
    """
    데이터셋을 감싸서, 원본 라벨 대신 무작위로 다른 라벨을 부여하는 역할
    Random-label 기법에 사용됨
    """
    def __init__(self, ds, num_classes=10):
        self.ds = ds; self.k = num_classes; self._labels = {}
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, y = self.ds[i]
        if i not in self._labels:
            nl = torch.randint(0, self.k, (1,)).item()
            while nl == y: nl = torch.randint(0, self.k, (1,)).item()
            self._labels[i] = nl
        return img, self._labels[i]

def _train_simple(model, loader, epochs, lr, device, momentum, weight_decay):
    """
    가장 기본적인 학습 루프. 언러닝 기법 내부에서 재사용하기 위해 만듦
    """
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()

def unlearn_finetune(orig, retain_loader, cfg):
    """Fine-tuning 기반 언러닝. 가장 간단한 baseline임"""
    m = copy.deepcopy(orig)
    # retain set만으로 모델을 추가 학습
    _train_simple(m, retain_loader, cfg["unlearn_epochs"], cfg["unlearn_lr"],
                  cfg["device"], cfg["momentum"], cfg["weight_decay"])
    return m

def unlearn_neggrad(orig, forget_loader, cfg):
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr_neggrad"])
    m.train()
    # forget set에 대해 학습을 진행하되, 손실 함수에 음수를 곱해 경사를 반대 방향으로 적용
    # 즉, 모델이 forget set을 '틀리도록' 학습하여 해당 정보를 잊게 만듦
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in forget_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad()
            loss = -crit(m(x), y) # 손실에 -1을 곱함
            loss.backward()
            opt.step()
    return m

def unlearn_l1_sparse(orig, retain_loader, cfg):
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"], momentum=cfg["momentum"])
    m.train()
    # Retain set으로 학습하되, 손실 함수에 L1 정규화 항을 추가
    # L1 정규화는 불필요한 가중치를 0으로 만드는 경향이 있어, 모델을 더 sparse하게 만들어 특정 데이터에 대한 의존성을 줄이는 효과를 기대함
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in retain_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad()
            l1 = sum(p.abs().sum() for p in m.parameters()) # 모든 파라미터의 절대값 합 (L1 norm)
            loss = crit(m(x), y) + cfg["l1_lambda"] * l1 # 원래 손실에 L1 항을 더함
            loss.backward(); opt.step()
    return m

def unlearn_neggrad_plus(orig, retain_loader, forget_loader, cfg):
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"])
    m.train(); r_iter = itertools.cycle(retain_loader)
    # Retain set에 대한 일반적인 학습(손실 최소화)과 Forget set에 대한 NegGrad(손실 최대화)를 동시에 수행
    for _ in range(cfg["unlearn_epochs"]):
        for fx, fy in forget_loader:
            rx, ry = next(r_iter)
            rx, ry = rx.to(cfg["device"]), ry.to(cfg["device"])
            fx, fy = fx.to(cfg["device"]), fy.to(cfg["device"])
            opt.zero_grad()
            # L_retain - α * L_forget
            loss = crit(m(rx), ry) - cfg["neggrad_plus_alpha"] * crit(m(fx), fy)
            loss.backward(); opt.step()
    return m

def unlearn_scrub(orig, retain_loader, forget_loader, cfg):
    """지식 증류(Knowledge Distillation)를 활용"""
    m = copy.deepcopy(orig); t_model = copy.deepcopy(orig).eval() # 현재 모델(m)과 별개로, 파라미터가 업데이트되지 않는 원본 모델(t_model)을 복사해 둠
    crit = nn.CrossEntropyLoss(); kld = nn.KLDivLoss(reduction="batchmean") # KL-Divergence Loss 사용
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"])
    m.train(); r_iter = itertools.cycle(retain_loader)
    # Retain set에 대해서는 올바르게 예측하도록 학습하고 (CrossEntropyLoss),
    # Forget set에 대해서는 현재 모델(m)의 예측 분포가 원본 모델(t_model)의 예측 분포와 '달라지도록' 학습함 (KL-Divergence Loss)
    # 즉, "기억할 건 잘 기억하고, 잊을 건 원래의 예측에서 멀어져라"
    for _ in range(cfg["unlearn_epochs"]):
        for fx, _ in forget_loader:
            rx, ry = next(r_iter)
            rx, ry, fx = rx.to(cfg["device"]), ry.to(cfg["device"]), fx.to(cfg["device"])
            opt.zero_grad()
            loss = (1-cfg["scrub_alpha"]) * crit(m(rx), ry) \
                   - cfg["scrub_alpha"] * kld(torch.log_softmax(m(fx),1),
                                              torch.softmax(t_model(fx),1))
            loss.backward(); opt.step()
    return m

def unlearn_random_label(orig, forget_set, cfg):
    m = copy.deepcopy(orig)
    loader = DataLoader(RelabelDataset(forget_set), batch_size=cfg["batch_size"], shuffle=True)
    _train_simple(m, loader, cfg["unlearn_epochs"], cfg["unlearn_lr"],
                  cfg["device"], cfg["momentum"], cfg["weight_decay"])
    return m

def unlearn_random_label_fixed(orig, retain_set, forget_set, cfg):
    m = copy.deepcopy(orig)
    # Forget set의 라벨을 무작위로 틀리게 바꾼 뒤, 이 데이터로 모델을 학습시킴
    # 모델이 틀린 정보를 배우게 하여 기존의 올바른 정보를 덮어쓰게 만듦
    comb = ConcatDataset([retain_set, RelabelDataset(forget_set)])
    loader = DataLoader(comb, batch_size=cfg["batch_size"], shuffle=True)
    _train_simple(m, loader, cfg["unlearn_epochs"], cfg["unlearn_lr"],
                  cfg["device"], cfg["momentum"], cfg["weight_decay"])
    return m

def unlearn_salun(orig, forget_set, cfg):
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    # 1. Forget set에 대한 각 파라미터의 gradient의 절대값을 누적하여 Saliency Map을 계산
    # 경사가 크다는 것은 해당 파라미터가 forget set의 예측에 큰 영향을 미친다는 의미임
    sal = [torch.zeros_like(p) for p in m.parameters()]
    f_loader = DataLoader(forget_set, batch_size=cfg["batch_size"])
    m.train()
    for x, y in f_loader:
        x, y = x.to(cfg["device"]), y.to(cfg["device"])
        m.zero_grad(); loss = crit(m(x), y); loss.backward()
        for i, p in enumerate(m.parameters()):
            if p.grad is not None: sal[i] += p.grad.abs()
    
    # 2. Saliency가 높은 상위 N%의 파라미터만 업데이트하도록 mask 생성
    flat = torch.cat([s.flatten() for s in sal])
    k = max(1, int(len(flat) * cfg["salun_sparsity"]))
    th, _ = torch.kthvalue(flat, k)
    masks = [(s > th).float() for s in sal]
    rl = DataLoader(RelabelDataset(forget_set), batch_size=cfg["batch_size"], shuffle=True)
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"], momentum=cfg["momentum"])

    # 3. Random Label 기법과 유사하게 학습하되, 경사를 업데이트할 때 위에서 만든 마스크를 곱해준다
    # 즉, forget set과 관련 깊은 파라미터만 선택적으로 수정하여 효율성을 높임
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in rl:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad(); loss = crit(m(x), y); loss.backward()
            for i, p in enumerate(m.parameters()):
                if p.grad is not None: p.grad *= masks[i] # 마스크 적용
            opt.step()
    return m

def unlearn_salun_fixed(orig, retain_set, forget_set, cfg):
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    sal = [torch.zeros_like(p) for p in m.parameters()]
    f_loader = DataLoader(forget_set, batch_size=cfg["batch_size"])
    m.train()
    for x, y in f_loader:
        x, y = x.to(cfg["device"]), y.to(cfg["device"])
        m.zero_grad(); loss = crit(m(x), y); loss.backward()
        for i, p in enumerate(m.parameters()):
            if p.grad is not None: sal[i] += p.grad.abs()
    flat = torch.cat([s.flatten() for s in sal])
    k = max(1, int(len(flat) * cfg["salun_sparsity"]))
    th, _ = torch.kthvalue(flat, k)
    masks = [(s > th).float() for s in sal]
    comb = ConcatDataset([retain_set, RelabelDataset(forget_set)])
    loader = DataLoader(comb, batch_size=cfg["batch_size"], shuffle=True)
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"], momentum=cfg["momentum"])
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad(); loss = crit(m(x), y); loss.backward()
            for i, p in enumerate(m.parameters()):
                if p.grad is not None: p.grad *= masks[i]
            opt.step()
    return m
