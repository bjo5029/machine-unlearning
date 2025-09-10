import os
import copy
import numpy as np
import pandas as pd
import hashlib
import time
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import stats
import argparse
import importlib

from seeds import set_seed
from data_es import split_retain_forget, build_loaders
from model_train import get_model, train_model, evaluate_model
from methods import (
    unlearn_ft, unlearn_ft_l1, unlearn_ga, unlearn_neggrad_plus,
    unlearn_rl, unlearn_wfisher, unlearn_scrub,
)
from metrics import calculate_prediction_diff, calculate_mia_score

# --- Helper functions ---
def _delta_metrics(rF, rR, rT, uF, uR, uT):
    dF = uF - rF
    dR = abs(uR - rR)
    dT = abs(uT - rT)
    return dF, dR, dT

def original_cache_path(save_dir, cfg):
    tag = f"{cfg['arch']}_E{cfg['epochs']}_lr{cfg['lr']}_s{cfg['seed']}"
    return os.path.join(save_dir, f"original_{tag}.pth")

def retrain_cache_key(retain_idx_sorted, cfg):
    h = hashlib.sha1(np.asarray(retain_idx_sorted, dtype=np.int64).tobytes()).hexdigest()
    return f"retrain_{h}_{cfg['arch']}_E{cfg['epochs']}_lr{cfg['lr']}_s{cfg['seed']}"

def retrain_cache_path(save_dir, retain_idx_sorted, cfg):
    return os.path.join(save_dir, retrain_cache_key(retain_idx_sorted, cfg) + ".pth")

def load_or_train_original(train_ds, bs, device, cfg):
    pth = original_cache_path(cfg["model_save_dir"], cfg)
    model = get_model(device)
    if os.path.exists(pth) and not cfg.get('retrain_all', False):
        print(f"[LOAD] Original model from {pth}")
        model.load_state_dict(torch.load(pth, map_location=device))
    else:
        print("[TRAIN] Original Model")
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        train_model(model, loader, cfg["epochs"], cfg["lr"], device, cfg["momentum"], cfg["weight_decay"])
        torch.save(model.state_dict(), pth); print(f"[SAVE] {pth}")
    return model

def load_or_train_retrain(retain_loader, retain_idx_sorted, device, cfg):
    pth = retrain_cache_path(cfg["model_save_dir"], retain_idx_sorted, cfg)
    model = get_model(device)
    if os.path.exists(pth) and not cfg.get('retrain_all', False):
        print(f"[LOAD] Retrain from {pth}")
        model.load_state_dict(torch.load(pth, map_location=device))
    else:
        if retain_loader is None or len(retain_loader.dataset) == 0:
            print("[SKIP] Retrain is skipped as retain_loader is empty.")
            return None, None
        print(f"[TRAIN] Retrain on {len(retain_loader.dataset)} samples")
        train_model(model, retain_loader, cfg["epochs"], cfg["lr"], device, cfg["momentum"], cfg["weight_decay"])
        torch.save(model.state_dict(), pth)
        print(f"[SAVE] {pth}")
    return model, pth

class SequentialCurriculumLoader:
    def __init__(self, partition_datasets, batch_size):
        self.partition_datasets = partition_datasets
        self.partition_loaders = []
        for ds in self.partition_datasets:
            self.partition_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
        
        if self.partition_datasets:
            self.dataset = self.partition_datasets[0].dataset

        self.total_batches = sum(len(loader) for loader in self.partition_loaders)

    def __iter__(self):
        for loader in self.partition_loaders:
            yield from loader

    def __len__(self):
        return self.total_batches

# --- Main Experiment Function ---
def run_experiment(cfg, train_ds, train_eval_ds, test_ds, orig_model):
    set_seed(cfg["seed"])
    device = cfg["device"]; bs = cfg["batch_size"]

    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    full_train_eval_loader = DataLoader(train_eval_ds, batch_size=bs, shuffle=False)
    
    forget_partitions = cfg['precomputed_forget_partitions']
    retain_partitions = cfg.get('precomputed_retain_partitions', [])
    
    total_forget_indices = np.concatenate(forget_partitions) if forget_partitions and len(forget_partitions[0]) > 0 else np.array([])

    unlearning_methods = {
        "FT": unlearn_ft, "FT_l1": unlearn_ft_l1, "GA": unlearn_ga,
        "NG": unlearn_neggrad_plus, "RL": unlearn_rl, "Wfisher": unlearn_wfisher,
        "SCRUB": unlearn_scrub,
    }
    all_results = []

    for method_name, unlearn_fn in unlearning_methods.items():
        print(f"\n===== Running Method: {method_name} =====")
        model_u = copy.deepcopy(orig_model)
        
        method_config = cfg.copy()
        if method_name in cfg.get("method_params", {}):
            method_config.update(cfg["method_params"][method_name])
        method_config['unlearn'] = method_name

        granularity = cfg.get("unlearning_granularity", "stage")
        is_paired_method = cfg.get("use_retain_ordering", False) and method_name in ["NG", "SCRUB"]

        if granularity == 'stage':
            cumulative_forget_indices = np.array([], dtype=np.int64)
            # ▼▼▼▼▼ [수정] 최종 Retain Set을 루프 시작 전에 미리 정의 ▼▼▼▼▼
            final_retain_idx, _ = split_retain_forget(len(train_ds), total_forget_indices)
            
            for stage_idx, S_forget_stage in enumerate(forget_partitions):
                stage = stage_idx + 1
                
                if is_paired_method and stage_idx < len(retain_partitions):
                    S_retain_stage = retain_partitions[stage_idx]
                    print(f"\n[UNLEARN PAIR] Stage {stage}: |Forget|={len(S_forget_stage)}, |Retain|={len(S_retain_stage)}")
                    r_train_unl, f_train_unl, _, _ = build_loaders(train_ds, train_eval_ds, S_retain_stage, S_forget_stage, bs)
                    loaders_for_unlearning = {"retain": r_train_unl, "forget": f_train_unl, "test": test_loader}
                else:
                    # ▼▼▼▼▼ [수정] 기본 방식: Forget은 해당 스테이지 파티션만, Retain은 미리 정의한 최종 Set 사용 ▼▼▼▼▼
                    forget_idx_unl = S_forget_stage
                    retain_idx_unl = final_retain_idx
                    
                    print(f"\n[UNLEARN] Stage {stage}: |Forget|={len(forget_idx_unl)}, |Retain|={len(retain_idx_unl)} (fixed)")
                    r_train_unl, f_train_unl, _, _ = build_loaders(train_ds, train_eval_ds, retain_idx_unl, forget_idx_unl, bs)
                    loaders_for_unlearning = {"retain": r_train_unl, "forget": f_train_unl, "test": test_loader}
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                model_u = unlearn_fn(model_u, loaders_for_unlearning, method_config)
                
                cumulative_forget_indices = np.unique(np.concatenate([cumulative_forget_indices, S_forget_stage]))
                eval_retain_idx, eval_forget_idx = split_retain_forget(len(train_ds), cumulative_forget_indices)
                r_train_eval, _, r_eval, f_eval = build_loaders(train_ds, train_eval_ds, eval_retain_idx, eval_forget_idx, bs, shuffle_train=False)
                
                retr_model, _ = load_or_train_retrain(r_train_eval, np.sort(eval_retain_idx), device, cfg)
                
                if retr_model is not None:
                    rF = evaluate_model(retr_model, f_eval, device); rR = evaluate_model(retr_model, r_eval, device); rT = evaluate_model(retr_model, test_loader, device)
                    uF = evaluate_model(model_u, f_eval, device); uR = evaluate_model(model_u, r_eval, device); uT = evaluate_model(model_u, test_loader, device)
                    dF, dR, dT = _delta_metrics(rF, rR, rT, uF, uR, uT); mia = calculate_mia_score(model_u, r_train_eval, r_eval, f_eval, test_loader, device); pdiff = calculate_prediction_diff(model_u, retr_model, full_train_eval_loader, device)
                    print(f"  (Eval) {method_name:>10} | S{stage} | Ftot={len(cumulative_forget_indices):>4d} | ΔF:{dF:+5.2f} ΔR:{dR:5.2f} ΔT:{dT:5.2f} | MIA:{mia:.4f} PredDiff:{pdiff:.2f}%")
                    all_results.append({"method": method_name, "stage": stage, "forget_total": len(cumulative_forget_indices), "Retain_F": rF, "Retain_R": rR, "Retrain_T": rT, "Unlearn_F": uF, "Unlearn_R": uR, "Unlearn_T": uT, "ΔF": dF, "ΔR": dR, "ΔT": dT, "MIA": mia, "PredDiff(%)": pdiff})

        elif granularity in ['batch', 'sample']:
            print(f"[UNLEARN MODE] {granularity.capitalize()}-wise curriculum for {method_name}")
            
            if cfg.get("use_retain_ordering", False):
                print("[INFO] Applying sequential curriculum to Retain Set.")
                if granularity == 'batch':
                    retain_subset_datasets = [Subset(train_ds, p) for p in retain_partitions]
                    retain_loader = SequentialCurriculumLoader(retain_subset_datasets, bs)
                else: 
                    sorted_retain_indices = retain_partitions[0]
                    retain_loader = DataLoader(Subset(train_ds, sorted_retain_indices), batch_size=bs, shuffle=False)
            else:
                print("[INFO] Using randomly shuffled Retain Set.")
                full_retain_idx, _ = split_retain_forget(len(train_ds), total_forget_indices)
                retain_loader = DataLoader(Subset(train_ds, full_retain_idx), batch_size=bs, shuffle=True)

            if granularity == 'batch':
                forget_subset_datasets = [Subset(train_ds, p) for p in forget_partitions]
                forget_loader = SequentialCurriculumLoader(forget_subset_datasets, bs)
            else: 
                sorted_forget_indices = forget_partitions[0]
                forget_loader = DataLoader(Subset(train_ds, sorted_forget_indices), batch_size=bs, shuffle=False)

            loaders = {"retain": retain_loader, "forget": forget_loader, "test": test_loader}
            model_u = unlearn_fn(model_u, loaders, method_config)

            final_retain_idx, final_forget_idx = split_retain_forget(len(train_ds), total_forget_indices)
            r_train_eval, _, r_eval, f_eval = build_loaders(train_ds, train_eval_ds, final_retain_idx, final_forget_idx, bs, shuffle_train=False)
            retr_model, _ = load_or_train_retrain(r_train_eval, np.sort(final_retain_idx), device, cfg)
            
            rF = evaluate_model(retr_model, f_eval, device); rR = evaluate_model(retr_model, r_eval, device); rT = evaluate_model(retr_model, test_loader, device)
            uF = evaluate_model(model_u, f_eval, device); uR = evaluate_model(model_u, r_eval, device); uT = evaluate_model(model_u, test_loader, device)
            dF, dR, dT = _delta_metrics(rF, rR, rT, uF, uR, uT); mia = calculate_mia_score(model_u, r_train_eval, r_eval, f_eval, test_loader, device); pdiff = calculate_prediction_diff(model_u, retr_model, full_train_eval_loader, device)
            print(f"  (Final Eval) {method_name:>10} | Ftot={len(final_forget_idx):>4d} | ΔF:{dF:+5.2f} ΔR:{dR:5.2f} ΔT:{dT:5.2f} | MIA:{mia:.4f} PredDiff:{pdiff:.2f}%")
            all_results.append({"method": method_name, "stage": "final", "forget_total": len(final_forget_idx), "Retain_F": rF, "Retrain_R": rR, "Retrain_T": rT, "Unlearn_F": uF, "Unlearn_R": uR, "Unlearn_T": uT, "ΔF": dF, "ΔR": dR, "ΔT": dT, "MIA": mia, "PredDiff(%)": pdiff})

    df = pd.DataFrame(all_results)
    print("\n===== Full Results =====")
    print(df.to_string(index=False))
    
    csv_filename = cfg.get("results_csv_filename", "experiment_results.csv")
    csv_path = os.path.join(cfg["model_save_dir"], csv_filename)
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    from data_es import define_forget_set, partition_forget_set, partition_retain_set, load_cifar10_with_train_eval, _embed_all, split_retain_forget
    from model_train import load_or_train_original

    print("This script is now intended to be run via run_all_conditions.py for full experiments.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str, default='config')
    args = parser.parse_args()
    config_module = importlib.import_module(args.config_module)
    CONFIG = config_module.CONFIG
    
    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval(CONFIG["data_root"])
    orig_model = load_or_train_original(train_ds, CONFIG['batch_size'], CONFIG['device'], CONFIG)
    
    global_mu = None
    if CONFIG.get("forget_partitioning_method") == 'es' or CONFIG.get("retain_partitioning_method") == 'es':
        all_embs = _embed_all(orig_model, train_eval_ds, CONFIG['device'], CONFIG['batch_size'])
        global_mu = all_embs.mean(0)
    
    total_forget_indices = define_forget_set(train_ds, CONFIG)
    CONFIG['precomputed_forget_partitions'] = partition_forget_set(total_forget_indices, train_eval_ds, orig_model, CONFIG, global_mu=global_mu)
    if CONFIG.get("use_retain_ordering", False):
        initial_retain_idx, _ = split_retain_forget(len(train_ds), total_forget_indices)
        CONFIG['precomputed_retain_partitions'] = partition_retain_set(initial_retain_idx, train_eval_ds, orig_model, CONFIG, global_mu=global_mu)

    run_experiment(CONFIG, train_ds, train_eval_ds, test_ds, orig_model)
