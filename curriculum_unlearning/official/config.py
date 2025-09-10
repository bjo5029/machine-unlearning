# config.py (최종 단순화 버전)
import torch
import os

CONFIG = {
    # --- 기본 설정 ---
    "data_root": "./data",
    "model_save_dir": "saved_models",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "cifar10",
    "arch": "resnet18",
    "batch_size": 256,
    "epochs": 1,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "unlearn_epochs": 10,
    "unlearn_lr": 0.01,
    "memorization_score_path": "./estimates_results.npz",
    "num_to_forget": 5000,
    "forget_class": 0,
    "input_size": 32,

    # ===================== GPU별 실행 분배용 ==================================
    # 이 리스트에 포함된 partitioning_method만 해당 GPU에서 실행됩니다.
    "methods_to_run_on_this_gpu": ["memorization", "es", "c_proxy", "random"],

    # ===================== 각 메소드별 상세 파라미터 ===========================
    # "method_params": {
    #     "FT": { "unlearn_epochs": 10, "unlearn_lr": 0.01 },
    #     "GA": { "unlearn_epochs": 10, "unlearn_lr": 1e-4 },
    #     "NG": { "unlearn_epochs": 5, "unlearn_lr": 0.01, "alpha": 0.9 },
    #     "RL": { "unlearn_epochs": 10, "unlearn_lr": 0.01, "decreasing_lr": "5,8" },
    #     "Wfisher": { "alpha": 10.0 },
    #     "SCRUB": { "unlearn_epochs": 10, "kd_T": 4.0, "gamma": 1.0, "beta": 1.0, "msteps": 5, "decreasing_lr": "5,8" }
    # }
    "method_params": {
        "FT": { "unlearn_epochs": 1, "unlearn_lr": 0.01 },
        "GA": { "unlearn_epochs": 1, "unlearn_lr": 1e-4 },
        "NG": { "unlearn_epochs": 1, "unlearn_lr": 0.01, "alpha": 0.9 },
        "RL": { "unlearn_epochs": 1, "unlearn_lr": 0.01, "decreasing_lr": "5,8" },
        "Wfisher": { "alpha": 10.0 },
        "SCRUB": { "unlearn_epochs": 1, "kd_T": 4.0, "gamma": 1.0, "beta": 1.0, "msteps": 5, "decreasing_lr": "5,8" }
    }
}
