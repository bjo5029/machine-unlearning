# config.py: 하이퍼파라미터, 경로, 시드 등 설정.

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
    "gpu": 0,                      # <-- 이 라인 추가 (사용할 GPU 인덱스)
    "imagenet_arch": False, 

    # --- Wandb 설정 ---
    "wandb_project": "machine-unlearning",
    "wandb_entity": None,  # wandb 계정 ID (없으면 None)
    "wandb_mode": "disabled",  # "online" 또는 "disabled"
    "wandb_group_name": None,
    "wandb_run_id": None,

    # --- 데이터 및 모델 설정 ---
    "dataset": "cifar10",  # 새 코드에서 사용
    "arch": "resnet18",    # 새 코드에서 사용
    "batch_size": 128,     # 새 코드 호환성을 위해 128 추천
    "num_classes": 10,     # 새 코드에서 사용

    # --- 원본/재학습 모델 학습 설정 ---
    "epochs": 1, # 코드 확인하고 다시 30으로 
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    # --- 언러닝 공통 설정 ---
    "unlearn_epochs": 1, # 코드 확인하고 다시 10으로
    "unlearn_lr": 0.01,
    "no_l1_epochs": 5,
    
    # --- 데이터 분할 설정 ---
    "forget_set_definition": "random", 
    "num_to_forget": 5000,
    "forget_class": 0,
    "forget_partitioning_method": "memorization",
    "memorization_score_path": "./estimates_results.npz",

    # --- impl.py 호환성을 위해 추가하는 설정값들 ---
    "group_index": 0,
    "mem_proxy": None,
    "mem": "GC", # 또는 다른 기본값

    # --- 새로운 언러닝 기법별 하이퍼파라미터 ---
    "alpha": 0.5,                  # NegGrad, Wfisher, FT_l1 등에서 사용
    "warmup": 0,                   # 에폭 초반 LR을 서서히 올리는 구간
    "print_freq": 100,             # 로그 출력 빈도
    "decreasing_lr": "5,8",        # LR 감소 스케줄
    
    # SCRUB 관련
    "kd_T": 4.0,
    "gamma": 1.0,
    "beta": 1.0,
    "msteps": 5,

    # Pruning 관련
    "prune_type": "rewind_lt",
    "rewind_epoch": 5,
    "rewind_pth": os.path.join("saved_models", "original_model.pth"),

    # Wfisher, Surgical Finetuning 관련
    "surgical": False,
    "choice": ['layer4.1.conv2', 'linear'],
}
