import torch

CONFIG = {
    "run_training": True,              # True: 학습+저장 / False: 저장 불러와서 평가만
    "model_save_dir": "saved_models",  # 스크린샷 구조에 맞춰 루트에 saved_models
    "num_runs": 3,
    "epochs": 30,
    "unlearn_epochs": 10,
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
}
