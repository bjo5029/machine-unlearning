# run_experiment.py (수정본)

import os, copy, numpy as np, pandas as pd, hashlib, time
from scipy import stats
import torch
from torch.utils.data import Subset

# from config import CONFIG  # <--- 이 라인을 삭제하거나 주석 처리합니다.
from seeds import set_seed
from data_es import (
    load_cifar10_with_train_eval, split_retain_forget, build_loaders,
    define_forget_set, partition_forget_set, hash_indices
)
from model_train import get_model, train_model, evaluate_model
from methods import (
    unlearn_ft, unlearn_ft_l1, unlearn_ga, unlearn_neggrad_plus,
    unlearn_rl, unlearn_wfisher, unlearn_scrub,
)
from metrics import calculate_prediction_diff, calculate_mia_score

def _delta_metrics(rF, rR, rT, uF, uR, uT):
    dF = uF - rF; dR = abs(uR - rR); dT = abs(uT - rT)
    return dF, dR, dT

def original_cache_path(save_dir, cfg):
    arch = cfg.get("arch", "model")
    tag = f"{arch}_E{cfg['epochs']}_lr{cfg['lr']}_m{cfg['momentum']}_wd{cfg['weight_decay']}_s{cfg['seed']}"
    return os.path.join(save_dir, f"original_{tag}.pth")

def retrain_cache_key(retain_idx_sorted, cfg):
    arch = cfg.get("arch", "model")
    h = hashlib.sha1(np.asarray(retain_idx_sorted, dtype=np.int64).tobytes()).hexdigest()
    return f"retrain_{h}_{arch}_E{cfg['epochs']}_lr{cfg['lr']}_m{cfg['momentum']}_wd{cfg['weight_decay']}_s{cfg['seed']}"

def retrain_cache_path(save_dir, retain_idx_sorted, cfg):
    return os.path.join(save_dir, retrain_cache_key(retain_idx_sorted, cfg) + ".pth")

def load_or_train_original(train_ds, bs, device, cfg):
    pth = original_cache_path(cfg["model_save_dir"], cfg)
    model = get_model(device)
    if os.path.exists(pth):
        print(f"[LOAD] Original model from {pth}")
        model.load_state_dict(torch.load(pth, map_location=device))
    else:
        print("[TRAIN] Original Model")
        full_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
        train_model(model, full_train_loader, cfg["epochs"], cfg["lr"], device, cfg["momentum"], cfg["weight_decay"])
        torch.save(model.state_dict(), pth)
        print(f"[SAVE] {pth}")
    return model

def load_or_train_retrain(retain_loader, retain_idx_sorted, device, cfg):
    pth = retrain_cache_path(cfg["model_save_dir"], retain_idx_sorted, cfg)
    model = get_model(device)
    if os.path.exists(pth):
        print(f"[LOAD] Retrain from {pth}")
        model.load_state_dict(torch.load(pth, map_location=device))
    else:
        print(f"[TRAIN] Retrain on {len(retain_loader.dataset)} samples")
        train_model(model, retain_loader, cfg["epochs"], cfg["lr"], device, cfg["momentum"], cfg["weight_decay"])
        torch.save(model.state_dict(), pth)
        print(f"[SAVE] {pth}")
    return model, pth

# 함수 정의를 def run_experiment(): 에서 def run_experiment(cfg): 로 변경합니다.
def run_experiment(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["model_save_dir"], exist_ok=True)
    device = cfg["device"]; bs = cfg["batch_size"]

    # 이하 파일 내 모든 'CONFIG'를 'cfg'로 변경합니다.
    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval(cfg["data_root"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)
    val_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)
    full_train_eval_loader = torch.utils.data.DataLoader(train_eval_ds, batch_size=bs, shuffle=False)

    orig_model = load_or_train_original(train_ds, bs, device, cfg)

    total_forget_indices = define_forget_set(train_ds, cfg)
    forget_partitions = partition_forget_set(total_forget_indices, train_eval_ds, orig_model, cfg)

    ordering = cfg.get("forget_partition_ordering", "easy_first")
    if ordering == "hard_first":
        forget_partitions.reverse()
        print("[INFO] Unlearning order: Hard first (high memorization -> low)")
    elif ordering == "random":
        np.random.shuffle(forget_partitions)
        print("[INFO] Unlearning order: Random")
    else:
        print("[INFO] Unlearning order: Easy first (low memorization -> high)")

    unlearning_methods = {
        "FT": unlearn_ft, "FT_l1": unlearn_ft_l1, "GA": unlearn_ga,
        "NG": unlearn_neggrad_plus, "RL": unlearn_rl, "Wfisher": unlearn_wfisher,
        "SCRUB": unlearn_scrub,
    }

    all_results = []
    retrain_state_cache = {}

    for method_name, unlearn_fn in unlearning_methods.items():
        print(f"\n===== Running Method: {method_name} =====")
        model_u = copy.deepcopy(orig_model)
        cumulative_forget_indices = np.array([], dtype=np.int64)

        for stage, S_forget in enumerate(forget_partitions, start=1):
            cumulative_forget_indices = np.unique(np.concatenate([cumulative_forget_indices, S_forget]))
            retain_idx, forget_idx = split_retain_forget(len(train_ds), cumulative_forget_indices)

            r_train, f_train, r_eval, f_eval = build_loaders(
                train_ds, train_eval_ds, retain_idx, forget_idx, bs, shuffle_train=True
            )
            loaders = {"retain": r_train, "forget": f_train, "test": test_loader, "val": val_loader}

            key = retrain_cache_key(np.sort(retain_idx), cfg)
            if key in retrain_state_cache:
                retr_model = get_model(device)
                retr_model.load_state_dict(retrain_state_cache[key])
                print(f"[CACHE] Retrain (in-memory) for stage {stage}")
            else:
                retr_model, retr_path = load_or_train_retrain(r_train, np.sort(retain_idx), device, cfg)
                retrain_state_cache[key] = copy.deepcopy(retr_model.state_dict())
            
            print(f"[UNLEARN] {method_name} stage {stage} (|forget_total|={len(cumulative_forget_indices)})")
            
            method_config = cfg.copy()
            if method_name in cfg.get("method_params", {}):
                method_specific_params = cfg["method_params"][method_name]
                method_config.update(method_specific_params)
                print(f"  > Applied specific params for {method_name}: {method_specific_params}")

            method_config['unlearn'] = method_name
            model_u = unlearn_fn(model_u, loaders, method_config)

            rF = evaluate_model(retr_model, f_eval, device); rR = evaluate_model(retr_model, r_eval, device)
            rT = evaluate_model(retr_model, test_loader, device); uF = evaluate_model(model_u, f_eval, device)
            uR = evaluate_model(model_u, r_eval, device); uT = evaluate_model(model_u, test_loader, device)

            dF, dR, dT = _delta_metrics(rF, rR, rT, uF, uR, uT)
            mia   = calculate_mia_score(model_u, r_train, r_eval, f_eval, test_loader, device)
            pdiff = calculate_prediction_diff(model_u, retr_model, full_train_eval_loader, device)

            print(f"{method_name:>10} | S{stage} | Ftot={len(cumulative_forget_indices):>4d} | "
                  f"Ret F/R/T: {rF:5.2f}/{rR:5.2f}/{rT:5.2f} | "
                  f"Unl F/R/T: {uF:5.2f}/{uR:5.2f}/{uT:5.2f} | "
                  f"ΔF:{dF:+5.2f} ΔR:{dR:5.2f} ΔT:{dT:5.2f} | "
                  f"MIA:{mia:.4f} PredDiff:{pdiff:.2f}%")

            all_results.append({
                "method": method_name, "stage": stage, "forget_total": len(cumulative_forget_indices),
                "Retrain_F": rF, "Retrain_R": rR, "Retrain_T": rT,
                "Unlearn_F": uF, "Unlearn_R": uR, "Unlearn_T": uT,
                "ΔF": dF, "ΔR": dR, "ΔT": dT, "MIA": mia, "PredDiff(%)": pdiff
            })

    df = pd.DataFrame(all_results)
    print("\n===== Full Results =====")
    print(df.to_string(index=False))
    
    default_csv_name = "experiment_results.csv"
    csv_filename = cfg.get("results_csv_filename", default_csv_name)
    csv_path = os.path.join(cfg["model_save_dir"], csv_filename)
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

# if __name__ == "__main__": 이 부분은 직접 실행할 때만 필요
