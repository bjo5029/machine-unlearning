# methods.py (어댑터 역할의 새로운 버전)

import torch
import torch.nn as nn
from argparse import Namespace
import copy

# unlearn_methods 디렉토리에서 get_unlearn_method 함수를 임포트
from unlearn_methods import get_unlearn_method

def _create_compatible_args(cfg_dict):
    """cfg 딕셔셔리를 외부 코드가 요구하는 args 객체로 변환하고 호환성을 맞춰줍니다."""
    args = Namespace(**cfg_dict)

    # --- 이름 변환 (호환성 레이어) ---
    # data_es.py와 config.py는 'forget_class'를 사용 -> impl.py는 'class_to_replace'를 기대
    args.class_to_replace = getattr(args, 'forget_class', None)
    
    # data_es.py와 config.py는 'num_to_forget'을 사용 -> impl.py는 'num_indexes_to_replace'를 기대
    args.num_indexes_to_replace = getattr(args, 'num_to_forget', None)
    
    # getattr(args, 'key', default_value)는 args에 key가 없어도 에러를 내지 않고 기본값을 사용하게 해줍니다.
    
    return args

def _run_iterative_unlearn(unlearn_fn, model, loaders, cfg_dict):
    """
    @iterative_unlearn 데코레이터가 붙은 함수들을 실행하기 위한 래퍼.
    옵티마이저, 스케줄러 등을 설정하고 unlearn_fn을 호출합니다.
    """
    # 1. cfg 딕셔너리를 args 객체로 변환
    args = _create_compatible_args(cfg_dict)
    
    # impl.py의 @iterative_unlearn이 모델 전체를 학습시키므로,
    # 우리는 여기서 루프를 돌 필요 없이 함수를 한 번만 호출하면 됩니다.
    # @iterative_unlearn 내부에서 unlearn_epochs만큼 루프가 실행됩니다.
    unlearn_fn(
        data_loaders=loaders,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        args=args,
        mask=None
    )
    return model

def _run_oneshot_unlearn(unlearn_fn, model, loaders, cfg_dict):
    """
    Wfisher처럼 단발성으로 실행되는 언러닝 함수들을 위한 래퍼.
    """
    args = _create_compatible_args(cfg_dict)
    
    # 이 함수들은 내부 루프가 없으므로, 그냥 한 번 호출합니다.
    model = unlearn_fn(
        data_loaders=loaders,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        args=args,
        mask=None
    )
    return model

# --- 각 언러닝 기법에 대한 호출 함수 정의 ---

def unlearn_ft(model, loaders, cfg):
    """Fine-tuning Wrapper"""
    m = copy.deepcopy(model)
    # unlearn_methods/__init__.py 에서 'FT'에 해당하는 함수를 가져옴
    unlearn_method = get_unlearn_method('FT')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_ft_l1(model, loaders, cfg):
    """Fine-tuning with L1 Wrapper"""
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('FT_l1')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_ga(model, loaders, cfg):
    """Gradient Ascent (NegGrad) Wrapper"""
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('GA')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_neggrad_plus(model, loaders, cfg):
    """Negative Gradient (Retain+Forget) Wrapper"""
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('NG')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_rl(model, loaders, cfg):
    """Random Label Wrapper"""
    m = copy.deepcopy(model)
    # RL_original.py의 RL_og 함수 사용
    unlearn_method = get_unlearn_method('RL_og')
    return _run_iterative_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_wfisher(model, loaders, cfg):
    """WoodFisher Unlearning Wrapper"""
    m = copy.deepcopy(model)
    unlearn_method = get_unlearn_method('wfisher')
    # Wfisher는 단발성(one-shot) 메소드
    return _run_oneshot_unlearn(unlearn_method, m, loaders, cfg)

def unlearn_scrub(model, loaders, cfg):
    """SCRUB Unlearning Wrapper"""
    # SCRUB은 model list를 받으므로 특별 처리
    student_model = copy.deepcopy(model)
    teacher_model = copy.deepcopy(model)
    module_list = nn.ModuleList([student_model, teacher_model])
    
    # cfg 딕셔너리를 args 객체로 변환
    args = _create_compatible_args(cfg)
    
    unlearn_method = get_unlearn_method('SCRUB')

    # --- 최종 수정된 부분 ---
    # 데코레이터가 요구하는 'model' 인자 자리에 `module_list` 전체를 전달
    # 이렇게 하면 데코레이터는 module_list.parameters()로 옵티마이저를 생성하고,
    # scrub 함수는 두 번째 인자로 module_list를 정상적으로 전달받게 됨
    unlearn_method(
        model=module_list,  # student_model 대신 module_list 전체를 전달
        data_loaders=loaders,
        criterion=nn.CrossEntropyLoss(),
        args=args,
        mask=None
    )
    return module_list[0] # 언러닝된 학생 모델 반환
