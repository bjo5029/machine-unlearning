# run_experiment.py: 새로운 실험 메인 스크립트.

import os, copy, numpy as np, pandas as pd, hashlib, time
from scipy import stats
import torch
from torch.utils.data import Subset

from config import CONFIG
from seeds import set_seed
from data_es import (
    load_cifar10_with_train_eval, split_retain_forget, build_loaders,
    define_forget_set, partition_forget_set, hash_indices
)
from model_train import get_model, train_model, evaluate_model
# 수정된 methods.py에서 새로운 래퍼 함수들을 import
from methods import (
    unlearn_ft,
    unlearn_ft_l1,
    unlearn_ga,
    unlearn_neggrad_plus,
    unlearn_rl,
    unlearn_wfisher,
    unlearn_scrub,
)
from metrics import calculate_prediction_diff, calculate_mia_score

def _delta_metrics(rF, rR, rT, uF, uR, uT):
    dF = uF - rF; dR = abs(uR - rR); dT = abs(uT - rT)
    return dF, dR, dT

def retrain_cache_path(save_dir, retain_idx_sorted):
    h = hashlib.sha1(np.asarray(retain_idx_sorted, dtype=np.int64).tobytes()).hexdigest()
    return os.path.join(save_dir, f"retrain_{h}.pth")

def run_experiment():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
    device = CONFIG["device"]; bs = CONFIG["batch_size"]

    # 1. 데이터 로드
    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval(CONFIG["data_root"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)
    # 평가용 로더들 추가
    val_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False) # 임시로 test_ds 사용
    full_train_eval_loader = torch.utils.data.DataLoader(train_eval_ds, batch_size=bs, shuffle=False)

    # 2. 원본 모델 학습
    orig_model = get_model(device)
    orig_pth = os.path.join(CONFIG["model_save_dir"], "original_model.pth")
    if CONFIG["run_training"] or not os.path.exists(orig_pth):
        print("[TRAIN] Original Model")
        full_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
        train_model(orig_model, full_train_loader, CONFIG["epochs"], CONFIG["lr"], device, CONFIG["momentum"], CONFIG["weight_decay"])
        torch.save(orig_model.state_dict(), orig_pth)
    else:
        orig_model.load_state_dict(torch.load(orig_pth, map_location=device))

    # 3. Forget Set 정의 및 분할
    total_forget_indices = define_forget_set(train_ds, CONFIG)
    forget_partitions = partition_forget_set(total_forget_indices, train_eval_ds, orig_model, CONFIG)

    # 4. 실험할 언러닝 기법 목록 정의 (새로운 함수로 교체)
    unlearning_methods = {
        # "FT": unlearn_ft,
        # "FT_l1": unlearn_ft_l1,
        # "GA": unlearn_ga,          # Gradient Ascent (기존 NegGrad와 유사)
        # "NG": unlearn_neggrad_plus, # NegGrad+
        # "RL": unlearn_rl,           # Random Label
        # "Wfisher": unlearn_wfisher,
        "SCRUB": unlearn_scrub,
    }

    all_results = []
    
    # --- 기법(Method) 루프 시작 ---
    for method_name, unlearn_fn in unlearning_methods.items():
        print(f"\n===== Running Method: {method_name} =====")
        
        model_u = copy.deepcopy(orig_model)
        cumulative_forget_indices = np.array([], dtype=np.int64)

        # --- 스테이지(Stage) 루프 시작 ---
        for stage, S_forget in enumerate(forget_partitions, start=1):
            cumulative_forget_indices = np.unique(np.concatenate([cumulative_forget_indices, S_forget]))
            retain_idx, forget_idx = split_retain_forget(len(train_ds), cumulative_forget_indices)

            r_train, f_train, r_eval, f_eval = build_loaders(
                train_ds, train_eval_ds, retain_idx, forget_idx, bs, shuffle_train=True
            )
            
            # 새 언러닝 코드들이 요구하는 포맷에 맞춘 loaders 딕셔너리
            loaders = {
                "retain": r_train, "forget": f_train, "test": test_loader, "val": val_loader
            }

            # Retrain 모델 (비교 기준선)
            retr_model = get_model(device)
            retr_path = retrain_cache_path(CONFIG["model_save_dir"], np.sort(retain_idx))
            if CONFIG["run_training"] or not os.path.exists(retr_path):
                print(f"[TRAIN] Retrain stage {stage} on {len(retain_idx)} samples")
                train_model(retr_model, r_train, CONFIG["epochs"], CONFIG["lr"], device, CONFIG["momentum"], CONFIG["weight_decay"])
                torch.save(retr_model.state_dict(), retr_path)
            else:
                retr_model.load_state_dict(torch.load(retr_path, map_location=device))

            # 언러닝 적용
            print(f"[UNLEARN] {method_name} stage {stage} (|forget_total|={len(cumulative_forget_indices)})")
            
            # CONFIG에 현재 메소드 이름을 동적으로 추가
            CONFIG['unlearn'] = method_name
            
            model_u = unlearn_fn(model_u, loaders, CONFIG)

            # 평가
            rF = evaluate_model(retr_model, f_eval, device)
            rR = evaluate_model(retr_model, r_eval, device)
            rT = evaluate_model(retr_model, test_loader, device)
            uF = evaluate_model(model_u, f_eval, device)
            uR = evaluate_model(model_u, r_eval, device)
            uT = evaluate_model(model_u, test_loader, device)

            dF, dR, dT = _delta_metrics(rF, rR, rT, uF, uR, uT)
            mia = calculate_mia_score(model_u, r_train, r_eval, f_eval, test_loader, device)
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
    
    csv_path = os.path.join(CONFIG["model_save_dir"], "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    run_experiment()
    