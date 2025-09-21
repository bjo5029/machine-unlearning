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
            all_indices = np.concatenate([ds.indices for ds in self.partition_datasets])
            original_full_dataset = self.partition_datasets[0].dataset
            self.dataset = Subset(original_full_dataset, all_indices)
        else:
            self.dataset = None

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
        "FT": unlearn_ft, "FT_l1": unlearn_ft_l1, 
        "GA": unlearn_ga,
        "NG": unlearn_neggrad_plus, "RL": unlearn_rl, "Wfisher": unlearn_wfisher,
        "SCRUB": unlearn_scrub,
    }
    all_results = []

    # 1. 벤치마크가 될 재학습 모델을 미리 로드하고, 목표 정확도를 계산합니다.
    #    (batch/sample 단위는 루프 밖에서 한 번만 수행)
    final_retain_idx, final_forget_idx = split_retain_forget(len(train_ds), total_forget_indices)
    r_train_eval_final, _, r_eval_final, f_eval_final = build_loaders(train_ds, train_eval_ds, final_retain_idx, final_forget_idx, bs, shuffle_train=False)
    
    retr_model, _ = load_or_train_retrain(r_train_eval_final, np.sort(final_retain_idx), device, cfg)
    if not retr_model:
        print("Retrain model could not be loaded or trained. Skipping all methods.")
        return

    # 목표 정확도 (Retrain Accuracies)
    rF = evaluate_model(retr_model, f_eval_final, device)
    rR = evaluate_model(retr_model, r_eval_final, device)
    rT = evaluate_model(retr_model, test_loader, device)
    print(f"\n[Benchmark] Retrain Accuracies -> Forget: {rF:.2f}%, Retain: {rR:.2f}%, Test: {rT:.2f}%")

    for method_name, unlearn_fn in unlearning_methods.items():
        print(f"\n===== Running Method: {method_name} =====")
        
        granularity = cfg.get("unlearning_granularity", "stage")
        if granularity not in ['batch', 'sample']:
            print(f"Skipping method {method_name} for unsupported granularity '{granularity}' in this mode.")
            continue

        # --- 2. 최적 에폭 탐색을 위한 변수 초기화 ---
        best_model_state = None
        best_epoch = -1
        min_metric = float('inf')
        
        method_config_base = cfg.copy()
        if method_name in cfg.get("method_params", {}):
            method_config_base.update(cfg["method_params"][method_name])
        
        method_config_base['unlearn'] = method_name
        
        max_epochs = method_config_base.get("unlearn_epochs", 1)
        model_u = copy.deepcopy(orig_model)

        # --- 3. 매 에폭마다 언러닝 & 평가를 반복하는 루프 ---
        print(f"[Find Best Epoch] Running up to {max_epochs} epochs to find the best model state...")
        for epoch in range(max_epochs):
            # 한 에폭만 학습하도록 설정 변경
            single_epoch_config = method_config_base.copy()
            single_epoch_config['unlearn_epochs'] = 1

            # 데이터 로더 준비 (매 에폭 동일한 로더 사용)
            if cfg.get("use_retain_ordering", False):
                if granularity == 'batch':
                    retain_subset_datasets = [Subset(train_ds, p) for p in retain_partitions]
                    retain_loader = SequentialCurriculumLoader(retain_subset_datasets, bs)
                else: 
                    sorted_retain_indices = retain_partitions[0]
                    retain_loader = DataLoader(Subset(train_ds, sorted_retain_indices), batch_size=bs, shuffle=False)
            else:
                full_retain_idx, _ = split_retain_forget(len(train_ds), total_forget_indices)
                retain_loader = DataLoader(Subset(train_ds, full_retain_idx), batch_size=bs, shuffle=True)

            if granularity == 'batch':
                forget_subset_datasets = [Subset(train_ds, p) for p in forget_partitions]
                forget_loader = SequentialCurriculumLoader(forget_subset_datasets, bs)
            else: 
                sorted_forget_indices = forget_partitions[0]
                forget_loader = DataLoader(Subset(train_ds, sorted_forget_indices), batch_size=bs, shuffle=False)
            
            loaders = {"retain": retain_loader, "forget": forget_loader, "test": test_loader}
            
            # 딱 1 에폭만 언러닝 수행
            model_u = unlearn_fn(model_u, loaders, single_epoch_config)

            # 성능 평가
            uF = evaluate_model(model_u, f_eval_final, device)
            uR = evaluate_model(model_u, r_eval_final, device)
            uT = evaluate_model(model_u, test_loader, device)

            # 제안하신 평가지표 계산 (Retrain과의 총 정확도 차이)
            current_metric = abs(uF - rF) + abs(uR - rR) + abs(uT - rT)
            
            print(f"  Epoch [{epoch+1}/{max_epochs}] | Accs (F/R/T): {uF:.2f}/{uR:.2f}/{uT:.2f} | Metric (lower is better): {current_metric:.4f}")

            # 최고 성능 모델 갱신
            if current_metric < min_metric:
                min_metric = current_metric
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model_u.state_dict())
        
        # 4. 가장 성능이 좋았던 모델의 상태로 복원
        if best_model_state:
            print(f"  > Best model found at epoch {best_epoch} with metric {min_metric:.4f}. Loading this model for final evaluation.")
            model_u.load_state_dict(best_model_state)
        else:
            print("  > No best model found, using model from last epoch.")

        # 5. 최종 선택된 모델로 나머지 상세 지표 계산
        uF = evaluate_model(model_u, f_eval_final, device)
        uR = evaluate_model(model_u, r_eval_final, device)
        uT = evaluate_model(model_u, test_loader, device)
        dF, dR, dT = _delta_metrics(rF, rR, rT, uF, uR, uT)
        mia = calculate_mia_score(model_u, r_train_eval_final, r_eval_final, f_eval_final, test_loader, device)
        pdiff = calculate_prediction_diff(model_u, retr_model, full_train_eval_loader, device)
        
        print(f"  (Final Eval) {method_name:>10} | Best Epoch: {best_epoch} | ΔF:{dF:+5.2f} ΔR:{dR:5.2f} ΔT:{dT:5.2f} | MIA:{mia:.4f} PredDiff:{pdiff:.2f}%")
        all_results.append({
            "method": method_name, "stage": f"final(best_ep{best_epoch})", "forget_total": len(final_forget_idx), 
            "Retain_F": rF, "Retrain_R": rR, "Retrain_T": rT, 
            "Unlearn_F": uF, "Unlearn_R": uR, "Unlearn_T": uT, 
            "ΔF": dF, "ΔR": dR, "ΔT": dT, "MIA": mia, "PredDiff(%)": pdiff
        })

    df = pd.DataFrame(all_results)
    if not df.empty:
        print("\n===== Full Results =====")
        print(df.to_string(index=False))
        
        csv_filename = cfg.get("results_csv_filename", "experiment_results.csv")
        csv_path = os.path.join(cfg["model_save_dir"], csv_filename)
        
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

# --- (파일의 나머지 부분은 기존과 동일) ---
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
    