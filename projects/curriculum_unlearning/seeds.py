# seeds.py: 시드 고정.

import random, numpy as np, torch

def set_seed(seed: int):
    """주어진 시드로 모든 주요 라이브러리의 난수 생성기 고정"""
    torch.manual_seed(seed)                 # PyTorch CPU 연산 시드 고정
    torch.cuda.manual_seed_all(seed)        # PyTorch 모든 GPU 연산 시드 고정
    np.random.seed(seed)                    # NumPy 시드 고정
    random.seed(seed)                       # 파이썬 내장 random 모듈 시드 고정
    torch.backends.cudnn.deterministic = True # cuDNN이 항상 동일한 알고리즘을 사용하도록 설정 (재현성 보장)
    torch.backends.cudnn.benchmark = False    # cuDNN의 자동 최적화 기능 비활성화 (재현성 보장)
