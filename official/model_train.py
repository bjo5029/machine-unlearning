# model_train.py: ResNet18 정의, 표준 학습/평가 루프.

import time, torch, torch.nn as nn, torch.optim as optim
from torchvision import models

def get_model(device, num_classes=10):
    """CIFAR-10용 ResNet-18 모델 생성"""
    m = models.resnet18(weights=None, num_classes=num_classes)
    return m.to(device)

def train_model(model, loader, epochs, lr, device, momentum, weight_decay, use_cosine=True):
    """학습 루프"""
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_cosine else None
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        if sch: sch.step()
        print(f"    Epoch {ep+1}/{epochs}  {time.time()-t0:.2f}s")

@torch.no_grad()
def evaluate_model(model, loader, device):
    """평가 함수"""
    model.eval(); tot = corr = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        tot += y.size(0); corr += (pred == y).sum().item()
    return 100.0 * corr / tot
