# seeds.py: 시드 고정, 재현 가능한 DataLoader 헬퍼.

import random, numpy as np, torch

def set_seed(seed: int):
    """
    주어진 시드로 모든 주요 라이브러리의 난수 생성기를 고정
    """
    torch.manual_seed(seed) # PyTorch의 CPU 연산에 대한 시드 고정
    torch.cuda.manual_seed_all(seed) # PyTorch의 모든 GPU 연산에 대한 시드 고정
    np.random.seed(seed) # NumPy 라이브러리의 시드 고정
    random.seed(seed) # 파이썬 내장 random 모듈의 시드 고정
    torch.backends.cudnn.deterministic = True # cuDNN이 항상 동일한 알고리즘을 사용하도록 설정 (속도는 약간 느려질 수 있음)
    torch.backends.cudnn.benchmark = False # cuDNN이 입력 크기에 따라 최적 알고리즘을 찾는 기능을 끔

def seeded_loader(dataset, batch_size, shuffle, seed):
    """
    시드가 고정된 DataLoader를 생성
    """
    g = torch.Generator(); g.manual_seed(seed) # 데이터 로딩 작업(특히 셔플링)에 사용할 별도의 난수 생성기 만듬
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=g, # 이 생성기(g)를 사용해 데이터 셔플
        worker_init_fn=lambda i: set_seed(seed + i) # 멀티프로세싱으로 데이터를 로드할 때 각 워커(프로세스)에 다른 시드를 부여해서 재현성 유지
    )
