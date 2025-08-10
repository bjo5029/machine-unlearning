import time, torch, torch.nn as nn, torch.optim as optim
from torchvision import models

def get_model(device, num_classes=10):
    m = models.resnet18(weights=None, num_classes=num_classes)
    return m.to(device)

def train_model(model, train_loader, epochs, lr, device, momentum, weight_decay, is_unlearning=False):
    crit = nn.CrossEntropyLoss()
    opt  = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sch  = None if is_unlearning else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        if sch: sch.step()
        print(f"    Epoch {ep+1}/{epochs} completed in {time.time()-t0:.2f}s")

def evaluate_model(model, loader, device):
    model.eval(); tot = corr = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            tot += y.size(0); corr += (pred == y).sum().item()
    return 100.0 * corr / tot

def calculate_prediction_diff(unlearned_model, retrained_model, loader, device):
    unlearned_model.eval(); retrained_model.eval()
    diff = total = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            p1 = unlearned_model(x).argmax(1)
            p2 = retrained_model(x).argmax(1)
            diff += (p1 != p2).sum().item()
            total += x.size(0)
    return 100.0 * diff / total
