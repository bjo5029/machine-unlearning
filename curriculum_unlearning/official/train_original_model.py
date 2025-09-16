# train_original_model.py (수정본)

import os
import argparse
import importlib
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader
import wandb

from data_es import load_cifar10_with_train_eval
from model_train import get_model, evaluate_model
from seeds import set_seed

# -------------------------
# Helpers
# -------------------------
def original_cache_path(save_dir: str, cfg: Dict):
    tag = f"{cfg['arch']}_E{cfg['epochs']}_lr{cfg['lr']}_s{cfg['seed']}"
    return os.path.join(save_dir, f"original_{tag}.pth")

def train_one_epoch(model, loader, device, optimizer, criterion, epoch, log_steps=50):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        if step % log_steps == 0:
            wandb.log({
                "train/step_loss": loss.item(),
                "train/step": (epoch - 1) * len(loader) + step
            })

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, device, split_name="eval"):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return {
        f"{split_name}/loss": running_loss / max(total, 1),
        f"{split_name}/acc": correct / max(total, 1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_module", type=str, default="config")
    parser.add_argument("--project", type=str, default="unlearning-original")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--resume", type=str, default="allow", choices=["allow", "never", "must"])
    parser.add_argument("--log_steps", type=int, default=50)
    args = parser.parse_args()

    CONFIG = importlib.import_module(args.config_module).CONFIG
    set_seed(CONFIG["seed"])

    device = CONFIG["device"]
    bs = CONFIG["batch_size"]

    os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

    # Data
    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval(CONFIG["data_root"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(train_eval_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # Model & Optim
    model = get_model(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["lr"],
        momentum=CONFIG.get("momentum", 0.9),
        weight_decay=CONFIG.get("weight_decay", 5e-4),
        nesterov=CONFIG.get("nesterov", False),
    )
    criterion = torch.nn.CrossEntropyLoss()

    # --- [추가] ---
    # Cosine Annealing 스케줄러를 추가합니다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    # ---

    # W&B init
    wandb_run_cfg = {
        "arch": CONFIG["arch"],
        "epochs": CONFIG["epochs"],
        "lr": CONFIG["lr"],
        "momentum": CONFIG.get("momentum", 0.9),
        "weight_decay": CONFIG.get("weight_decay", 5e-4),
        "batch_size": CONFIG["batch_size"],
        "seed": CONFIG["seed"],
        "device": str(CONFIG["device"]),
        "data_root": CONFIG["data_root"],
    }

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        mode=args.mode,
        resume=args.resume,
        config=wandb_run_cfg,
    )
    wandb.watch(model, log="all", log_graph=False)

    best_eval_acc = 0.0
    best_path = original_cache_path(CONFIG["model_save_dir"], CONFIG)

    # Optionally resume from cache if exists
    if os.path.exists(best_path) and not CONFIG.get("retrain_all", False):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt)
        # Evaluate existing model
        eval_metrics = evaluate(model, eval_loader, device, "eval")
        test_metrics = evaluate(model, test_loader, device, "test")
        wandb.log({**eval_metrics, **test_metrics, "epoch": 0})
        print(f"[LOAD] Pretrained original model found at {best_path}")
    else:
        print("[TRAIN] Start training original model")

        for epoch in range(1, CONFIG["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, epoch, args.log_steps)
            eval_metrics = evaluate(model, eval_loader, device, "eval")
            test_metrics = evaluate(model, test_loader, device, "test")
            elapsed = time.time() - t0

            # --- [추가] ---
            # 매 에폭이 끝날 때마다 스케줄러를 업데이트합니다.
            scheduler.step()
            # ---

            # --- [수정] ---
            # 스케줄러가 적용된 현재 학습률을 기록합니다.
            current_lr = scheduler.get_last_lr()[0]
            # ---

            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "train/lr": current_lr,
                **eval_metrics,
                **test_metrics,
                "time/epoch_sec": elapsed,
            })

            # Save best by eval acc
            if eval_metrics["eval/acc"] >= best_eval_acc:
                best_eval_acc = eval_metrics["eval/acc"]
                torch.save(model.state_dict(), best_path)
                wandb.log({"checkpoint/saved": epoch})
                print(f"[SAVE] {best_path} (epoch {epoch}, eval_acc={best_eval_acc:.4f})")

    # Final log of best checkpoint metrics
    model.load_state_dict(torch.load(best_path, map_location=device))
    final_eval = evaluate(model, eval_loader, device, "eval_best")
    final_test = evaluate(model, test_loader, device, "test_best")
    wandb.log({**final_eval, **final_test})
    print("[DONE] Best checkpoint re-evaluated and logged.")

if __name__ == "__main__":
    main()
    