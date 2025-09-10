import torch
import torch.nn as nn
from argparse import Namespace
import copy
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from unlearn_methods import get_unlearn_method

def _create_compatible_args(cfg_dict):
    """cfg 딕셔셔리를 외부 코드가 요구하는 args 객체로 완벽히 변환하고 호환성을 맞춰줍니다."""

    default_args = {
        # 데이터셋 기본 설정
        "num_classes": 10,
        
        # Wandb 설정
        "wandb_project": "machine-unlearning", "wandb_entity": None,
        "wandb_mode": "disabled", "wandb_group_name": None, "wandb_run_id": None,
        
        # 언러닝 공통 설정 및 기타 호환성 설정
        "no_l1_epochs": 5, # ▼▼▼▼▼ [수정] 이 라인을 추가했습니다. ▼▼▼▼▼
        "group_index": 0, "mem_proxy": None, "mem": "GC", "alpha": 1e-5,
        "salun_sparsity": 0.1, "warmup": 0, "print_freq": 100, 
        "decreasing_lr": "5,8", "kd_T": 4.0, "gamma": 1.0, "beta": 1.0, 
        "msteps": 5, "prune_type": "rewind_lt", "rewind_epoch": 5, 
        "rewind_pth": "saved_models/original_model.pth",
        "surgical": False, "choice": ['layer4.1.conv2', 'linear'],
        "gpu": 0, "imagenet_arch": False,

        # 저희 config에는 없지만 impl.py가 요구하는 값들
        "sequential": False, "shuffle": False, "unlearn_step": 0,
        "uname": "", "input_size": 32, "workers": 4, "no_aug": False,
        "indexes_to_replace": None,
    }
    
    default_args.update(cfg_dict)
    
    args = Namespace(**default_args)
    
    # --- 이름이 다른 변수들 최종 매핑 ---
    args.save_dir = args.model_save_dir
    args.class_to_replace = args.forget_class
    args.num_indexes_to_replace = args.num_to_forget
    if not hasattr(args, 'train_seed'): 
        args.train_seed = args.seed
            
    return args

def _run_iterative_unlearn(unlearn_fn, model, loaders, cfg_dict):
    args = _create_compatible_args(cfg_dict)
    unlearn_fn(data_loaders=loaders, model=model, criterion=nn.CrossEntropyLoss(), args=args, mask=None)
    return model

def _run_oneshot_unlearn(unlearn_fn, model, loaders, cfg_dict):
    args = _create_compatible_args(cfg_dict)
    model = unlearn_fn(data_loaders=loaders, model=model, criterion=nn.CrossEntropyLoss(), args=args, mask=None)
    return model

# --- 각 언러닝 기법에 대한 호출 함수 정의 (이하 동일) ---
def unlearn_ft(model, loaders, cfg):
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('FT')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_ft_l1(model, loaders, cfg):
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('FT_l1')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_ga(model, loaders, cfg):
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('GA')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_neggrad_plus(model, loaders, cfg):
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('NG')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_rl(model, loaders, cfg):
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('RL_og')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_wfisher(model, loaders, cfg):
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('wfisher')
    return _run_oneshot_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_scrub(model, loaders, cfg):
    student_model = copy.deepcopy(model)
    teacher_model = copy.deepcopy(model)
    module_list = nn.ModuleList([student_model, teacher_model])
    args = _create_compatible_args(cfg)
    unlearn_method = get_unlearn_method('SCRUB')
    unlearn_method(
        model=module_list,
        data_loaders=loaders,
        criterion=nn.CrossEntropyLoss(),
        args=args,
        mask=None
    )
    return module_list[0]
