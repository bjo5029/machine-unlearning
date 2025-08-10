# config.py: 하이퍼파라미터와 경로, 시드.

import torch

CONFIG = {
    "run_training": True,
    "model_save_dir": "saved_models",
    "num_runs": 1,
    "epochs": 30,
    "unlearn_epochs": 30,
    "batch_size": 256,
    "lr": 0.1,
    "unlearn_lr": 0.01,
    "unlearn_lr_neggrad": 1e-4,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "forget_set_size": 3000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "l1_lambda": 1e-5,
    "neggrad_plus_alpha": 0.2,
    "salun_sparsity": 0.5,
    "scrub_alpha": 0.5,
    "data_root": "../../data",
    "seed": 42,
}
