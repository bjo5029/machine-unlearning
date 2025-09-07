# run_experiment.py (AttributeError 수정된 최종본)

import os, copy, numpy as np, pandas as pd, hashlib, time
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import stats
import argparse
import importlib

from seeds import set_seed
from data_es import (
    load_cifar10_with_train_eval, split_retain_forget, build_loaders,
    define_forget_set, partition_forget_set, partition_retain_set, hash_indices
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
        print(f"[TRAIN] Retrain on {len(retain_loader.dataset)} samples")
        train_model(model, retain_loader, cfg["epochs"], cfg["lr"], device, cfg["momentum"], cfg["weight_decay"])
        torch.save(model.state_dict(), pth); print(f"[SAVE] {pth}")
    return model, pth

def create_interleaved_loader(datasets, batch_size):
    class InterleavedDataset(Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.indices = []
            for i, d in enumerate(self.datasets):
                self.indices.extend([(i, j) for j in range(len(d))])
            # [수정] .dataset 속성 추가
            if self.datasets:
                self.dataset = self.datasets[0].dataset
                
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            dataset_idx, sample_idx = self.indices[idx]
            return self.datasets[dataset_idx][sample_idx]

    interleaved_dataset = InterleavedDataset(datasets)
    return DataLoader(interleaved_dataset, batch_size=batch_size, shuffle=True)

def run_experiment(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["model_save_dir"], exist_ok=True)
    device = cfg["device"]; bs = cfg["batch_size"]

    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval(cfg["data_root"])
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    full_train_eval_loader = DataLoader(train_eval_ds, batch_size=bs, shuffle=False)
    orig_model = load_or_train_original(train_ds, bs, device, cfg)
    
    cfg["rewind_pth"] = original_cache_path(cfg["model_save_dir"], cfg)

    total_forget_indices = define_forget_set(train_ds, cfg)
    forget_partitions = partition_forget_set(total_forget_indices, train_eval_ds, orig_model, cfg)
    
    if cfg.get("use_retain_ordering", False):
        initial_retain_idx, _ = split_retain_forget(len(train_ds), total_forget_indices)
        retain_partitions = partition_retain_set(initial_retain_idx, train_eval_ds, orig_model, cfg)
    else:
        retain_partitions = []

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
            print(f"  > Applied specific params for {method_name}: {cfg['method_params'][method_name]}")
        method_config['unlearn'] = method_name

        granularity = cfg.get("unlearning_granularity", "stage")
        is_paired_method = cfg.get("use_retain_ordering", False) and method_name in ["NG", "SCRUB"]

        if granularity == 'stage':
            cumulative_forget_indices = np.array([], dtype=np.int64)
            num_stages = len(forget_partitions)
            
            for stage_idx in range(num_stages):
                stage = stage_idx + 1
                S_forget_stage = forget_partitions[stage_idx]
                
                if is_paired_method and stage_idx < len(retain_partitions):
                    S_retain_stage = retain_partitions[stage_idx]
                    print(f"\n[UNLEARN PAIR] Stage {stage}: |Forget|={len(S_forget_stage)}, |Retain|={len(S_retain_stage)}")
                    r_train_unl, f_train_unl, _, _ = build_loaders(train_ds, train_eval_ds, S_retain_stage, S_forget_stage, bs)
                    loaders_for_unlearning = {"retain": r_train_unl, "forget": f_train_unl, "test": test_loader}
                else:
                    cumulative_indices_for_unlearn = np.unique(np.concatenate([cumulative_forget_indices, S_forget_stage]))
                    retain_idx_unl, forget_idx_unl = split_retain_forget(len(train_ds), cumulative_indices_for_unlearn)
                    r_train_unl, f_train_unl, _, _ = build_loaders(train_ds, train_eval_ds, retain_idx_unl, forget_idx_unl, bs)
                    loaders_for_unlearning = {"retain": r_train_unl, "forget": f_train_unl, "test": test_loader}
                    print(f"\n[UNLEARN] Stage {stage}: |Forget Total|={len(cumulative_indices_for_unlearn)}")

                model_u = unlearn_fn(model_u, loaders_for_unlearning, method_config)
                
                cumulative_forget_indices = np.unique(np.concatenate([cumulative_forget_indices, S_forget_stage]))
                eval_retain_idx, eval_forget_idx = split_retain_forget(len(train_ds), cumulative_forget_indices)
                r_train_eval, _, r_eval, f_eval = build_loaders(train_ds, train_eval_ds, eval_retain_idx, eval_forget_idx, bs, shuffle_train=False)
                
                retr_model, _ = load_or_train_retrain(r_train_eval, np.sort(eval_retain_idx), device, cfg)

                rF = evaluate_model(retr_model, f_eval, device); rR = evaluate_model(retr_model, r_eval, device); rT = evaluate_model(retr_model, test_loader, device)
                uF = evaluate_model(model_u, f_eval, device); uR = evaluate_model(model_u, r_eval, device); uT = evaluate_model(model_u, test_loader, device)
                dF, dR, dT = _delta_metrics(rF, rR, rT, uF, uR, uT); mia = calculate_mia_score(model_u, r_train_eval, r_eval, f_eval, test_loader, device); pdiff = calculate_prediction_diff(model_u, retr_model, full_train_eval_loader, device)
                print(f"  (Eval) {method_name:>10} | S{stage} | Ftot={len(cumulative_forget_indices):>4d} | ΔF:{dF:+5.2f} ΔR:{dR:5.2f} ΔT:{dT:5.2f} | MIA:{mia:.4f} PredDiff:{pdiff:.2f}%")
                
                all_results.append({"method": method_name, "stage": stage, "forget_total": len(cumulative_forget_indices), "Retrain_F": rF, "Retrain_R": rR, "Retrain_T": rT, "Unlearn_F": uF, "Unlearn_R": uR, "Unlearn_T": uT, "ΔF": dF, "ΔR": dR, "ΔT": dT, "MIA": mia, "PredDiff(%)": pdiff})

        elif granularity in ['batch', 'sample']:
            print(f"[UNLEARN MODE] {granularity.capitalize()}-wise curriculum for {method_name}")
            full_retain_idx, _ = split_retain_forget(len(train_ds), total_forget_indices)
            retain_loader = DataLoader(Subset(train_ds, full_retain_idx), batch_size=bs, shuffle=True)
            
            if granularity == 'batch':
                forget_subset_datasets = [Subset(train_ds, p) for p in forget_partitions]
                forget_loader = create_interleaved_loader(forget_subset_datasets, bs)
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
            all_results.append({"method": method_name, "stage": "final", "forget_total": len(final_forget_idx), "Retrain_F": rF, "Retrain_R": rR, "Retrain_T": rT, "Unlearn_F": uF, "Unlearn_R": uR, "Unlearn_T": uT, "ΔF": dF, "ΔR": dR, "ΔT": dT, "MIA": mia, "PredDiff(%)": pdiff})

    df = pd.DataFrame(all_results)
    print("\n===== Full Results =====")
    print(df.to_string(index=False))
    
    default_csv_name = "experiment_results.csv"
    csv_filename = cfg.get("results_csv_filename", default_csv_name)
    csv_path = os.path.join(cfg["model_save_dir"], csv_filename)
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single unlearning experiment.")
    parser.add_argument('--config_module', type=str, default='config', 
                        help='The name of the configuration file to use (e.g., config)')
    args = parser.parse_args()

    print(f"Attempting to run a single experiment using configuration from '{args.config_module}.py'")
    
    try:
        config_module = importlib.import_module(args.config_module)
        CONFIG = config_module.CONFIG
        print("Configuration loaded successfully.")
        run_experiment(CONFIG)
    except ImportError:
        print(f"Error: Could not import configuration module '{args.config_module}'. Please ensure the file exists.")
    except AttributeError:
        print(f"Error: The configuration file '{args.config_module}.py' does not contain a 'CONFIG' dictionary.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        