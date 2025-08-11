import torch

CONFIG = {
    # I/O
    "data_root": "./data",
    "model_save_dir": "saved_models",   # 자동 생성
    "seed": 42,

    # Training
    "run_training": True,               # 캐시가 있더라도 다시 학습할지 여부
    "num_runs": 1,                      # 빠른 스모크 테스트 후 3으로
    "epochs": 30,                       # 원모델/재학습(retrain) 에폭
    "unlearn_epochs": 10,               # 스테이지당 언러닝 에폭
    "batch_size": 256,
    "lr": 0.1,
    "unlearn_lr": 0.01,
    "unlearn_lr_neggrad": 1e-4,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    # Partition
    "forget_set_size": 3000,            # 한 스테이지당 제거 수 (총 3스테이지 = 9000)
    "use_paper_es": True,               # True: paper-like ES 파티션, False: balanced 변형
    "balanced_bins": 5,                 # balanced 모드에서 confidence bins

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
