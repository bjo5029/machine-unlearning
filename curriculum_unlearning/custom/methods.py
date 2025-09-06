# methods.py: 언러닝 기법들.

import copy, torch, itertools
import torch.nn as nn
import torch.optim as optim

def _train_simple(model, loader, epochs, lr, device, momentum, weight_decay):
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()

def unlearn_finetune(orig, retain_loader, cfg):
    m = copy.deepcopy(orig)
    _train_simple(m, retain_loader, cfg["unlearn_epochs"], cfg["unlearn_lr"],
                  cfg["device"], cfg["momentum"], cfg["weight_decay"])
    return m

def unlearn_l1_sparse(orig, retain_loader, cfg):
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    m.train()
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in retain_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad()
            l1 = sum(p.abs().sum() for p in m.parameters())
            loss = crit(m(x), y) + 1e-5 * l1
            loss.backward(); opt.step()
    return m

def unlearn_neggrad(orig, forget_loader, cfg):
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr_neggrad"])
    m.train()
    for _ in range(cfg["unlearn_epochs"]):
        for x, y in forget_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            opt.zero_grad(); loss = -crit(m(x), y); loss.backward(); opt.step()
    return m

def unlearn_neggrad_plus(orig, retain_loader, forget_loader, cfg):
    """Retain 데이터로는 올바르게 학습하고(손실 최소화), Forget 데이터로는 반대로 학습하여(손실 최대화) 정보를 잊게 만듦"""
    m = copy.deepcopy(orig); crit = nn.CrossEntropyLoss()
    opt = optim.SGD(m.parameters(), lr=cfg["unlearn_lr"])
    m.train(); r_iter = itertools.cycle(retain_loader) # retain 로더를 무한 반복하도록 설정
    alpha = cfg.get("neggrad_plus_alpha", 0.2) # forget 손실에 곱해줄 가중치
    for _ in range(cfg["unlearn_epochs"]):
        for fx, fy in forget_loader: # forget 로더를 기준으로 루프를 돈다
            rx, ry = next(r_iter) # retain 로더에서 배치를 하나 가져옴
            rx, ry = rx.to(cfg["device"]), ry.to(cfg["device"])
            fx, fy = fx.to(cfg["device"]), fy.to(cfg["device"])
            opt.zero_grad()
            # 최종 손실 = (Retain 손실) - α * (Forget 손실)
            loss = crit(m(rx), ry) - alpha * crit(m(fx), fy)
            loss.backward(); opt.step() # 계산된 경사로 파라미터 업데이트
    return m
