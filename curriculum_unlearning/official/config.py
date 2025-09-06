# config.py (최종 수정본)

import torch
import os

CONFIG = {
    # --- 기본 설정 ---
    "data_root": "./data",
    "model_save_dir": "saved_models",
    "seed": 42,
    "run_training": True,
    "num_runs": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gpu": 0,
    "imagenet_arch": False, 

    # --- Wandb 설정 ---
    "wandb_project": "machine-unlearning",
    "wandb_entity": None,
    "wandb_mode": "disabled",
    "wandb_group_name": None,
    "wandb_run_id": None,

    # --- 데이터 및 모델 설정 ---
    "dataset": "cifar10",
    "arch": "resnet18",
    "batch_size": 256,
    "num_classes": 10,

    # --- 원본/재학습 모델 학습 설정 ---
    "epochs": 30, 
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    # --- 언러닝 공통 설정 ---
    "unlearn_epochs": 10,
    "unlearn_lr": 0.01,
    "no_l1_epochs": 5,
    
    # --- 실험 조건 설정 ---
    "forget_set_definition": "class",
    "num_to_forget": 5000,
    "forget_class": 0,
    "forget_partitioning_method": "memorization",
    "memorization_score_path": "./estimates_results.npz",
    "forget_partition_ordering": "easy_first",
    
    # --- impl.py 호환성 등 기타 설정 ---
    "group_index": 0, "mem_proxy": None, "mem": "GC", "alpha": 1e-5, 
    "salun_sparsity": 0.1, "warmup": 0, "print_freq": 100, 
    "decreasing_lr": "5,8", "kd_T": 4.0, "gamma": 1.0, "beta": 1.0, 
    "msteps": 5, "prune_type": "rewind_lt", "rewind_epoch": 5, 
    "rewind_pth": os.path.join("saved_models", "original_model.pth"),
    "surgical": False, "choice": ['layer4.1.conv2', 'linear'],

    # ===============================================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 각 메소드별 상세 파라미터 정의 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # ===============================================================================
    "method_params": {
        # 각 메소드에 필요한 공통 파라미터(decreasing_lr 등)를 모두 추가
        "FT": {
            "unlearn_epochs": 10, "unlearn_lr": 0.01,
            "decreasing_lr": "5,8", "warmup": 0, "print_freq": 100
        },
        "FT_l1": {
            "unlearn_epochs": 10, "unlearn_lr": 0.005, "alpha": 1e-5,
            "decreasing_lr": "5,8", "warmup": 0, "print_freq": 100
        },
        "GA": {
            "unlearn_epochs": 10, "unlearn_lr": 1e-4,
            "decreasing_lr": "5,8", "warmup": 0, "print_freq": 100
        },
        "NG": {
            "unlearn_epochs": 5, "unlearn_lr": 0.01, "alpha": 0.9,
            "decreasing_lr": "2,4", "warmup": 0, "print_freq": 100 # NegGrad+는 에포크가 짧으므로 스케줄 조정
        },
        "RL": {
            "unlearn_epochs": 10, "unlearn_lr": 0.01,
            "decreasing_lr": "5,8", "warmup": 0, "print_freq": 100
        },
        "Wfisher": {
            "alpha": 10.0
        },
        "SCRUB": {
            "unlearn_epochs": 10, "kd_T": 4.0, "gamma": 1.0, "beta": 1.0, "msteps": 5,
            "decreasing_lr": "5,8", "warmup": 0, "print_freq": 100
        }
    }
}
