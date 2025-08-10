# seeds.py: 시드 고정, 재현 가능한 DataLoader 헬퍼.

import random, numpy as np, torch

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seeded_loader(dataset, batch_size, shuffle, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
        worker_init_fn=lambda i: set_seed(seed + i)
    )
