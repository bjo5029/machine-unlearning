# rum_bridge.py
from types import SimpleNamespace
import copy
import torch
import torch.nn as nn

# === RUM 함수 임포트 (파일명이 다르면 여기만 맞추세요) ===
from FT import FT as RUM_FT, FT_l1 as RUM_FT_L1           # retain 전용
from GA import GA as RUM_GA                               # forget 전용 (gradient ascent)
from neggrad import NG as RUM_NG                          # both (retain↓ + forget↑)
from scrub import SCRUB as RUM_SCRUB                      # both (KD 기반)
from Wfisher import Wfisher as RUM_WFISHER                # both (influence/WoodFisher)
from RL import RL as RUM_SALUN                            # both (README 기준 SalUn)
from RL_original import RL_og as RUM_RANDOM_LABEL         # both (랜덤 라벨)
# retrain은 당신 쪽에 이미 있으니 그대로 쓰면 됩니다.

# ---- 공통 args 어댑터 ----
def _make_args(cfg, **overrides):
    # cfg(dict) -> RUM이 기대하는 네이밍으로 변환
    return SimpleNamespace(
        unlearn_epochs = cfg.get("unlearn_epochs", 5),
        unlearn_lr     = cfg.get("unlearn_lr", 1e-2),
        warmup         = cfg.get("warmup", 0),
        no_l1_epochs   = cfg.get("no_l1_epochs", 0),
        alpha          = cfg.get("alpha", 0.0),   # NG/L1/WF 등에서 사용
        beta           = cfg.get("beta", 0.0),    # SCRUB
        gamma          = cfg.get("gamma", 0.0),   # SCRUB
        msteps         = cfg.get("msteps", 1),    # SCRUB
        kd_T           = cfg.get("kd_T", 1.0),    # SCRUB
        imagenet_arch  = cfg.get("imagenet_arch", False),
        print_freq     = cfg.get("print_freq", 50),
        num_classes    = cfg.get("num_classes", 10),
        wandb          = cfg.get("wandb", False), # RUM wandb_init을 끄고 싶으면 False
        **overrides
    )

def _criterion_optimizer(model, cfg):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get("unlearn_lr", 1e-2),
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 5e-4),
        nesterov=cfg.get("nesterov", False),
    )
    return crit, opt

# ---- DataLoader → RUM data_loaders dict 변환 ----
def _pack_loaders(retain_loader=None, forget_loader=None):
    d = {}
    if retain_loader is not None: d["retain"] = retain_loader
    if forget_loader is not None: d["forget"] = forget_loader
    return d

# ============ 래퍼들: 내 인터페이스(모델 사본 반환) ============

def unlearn_ft_rum(model, retain_loader, cfg, with_l1=False, mask=None):
    m = copy.deepcopy(model)   # 내 쪽은 사본 반환을 선호
    args = _make_args(cfg)
    data_loaders = _pack_loaders(retain_loader=retain_loader)
    crit, opt = _criterion_optimizer(m, cfg)

    for epoch in range(args.unlearn_epochs):
        if with_l1:
            # FT_l1은 내부에서 L1 감쇠 스케줄을 씀
            RUM_FT_L1(data_loaders, m, crit, opt, epoch, args, mask=mask)
        else:
            RUM_FT(data_loaders, m, crit, opt, epoch, args, mask=mask)
    return m

def unlearn_ga_rum(model, forget_loader, cfg, with_l1=False, mask=None):
    # GA는 forget 전용. (RUM_GA가 내부에서 with_l1 옵션을 받지 않는 구현이면 alpha=0으로 두거나 GA_l1을 따로 임포트하세요)
    m = copy.deepcopy(model)
    args = _make_args(cfg)
    data_loaders = _pack_loaders(forget_loader=forget_loader)
    crit, opt = _criterion_optimizer(m, cfg)

    for epoch in range(args.unlearn_epochs):
        RUM_GA(data_loaders, m, crit, opt, epoch, args, mask=mask)
    return m

def unlearn_ng_rum(model, retain_loader, forget_loader, cfg, mask=None):
    m = copy.deepcopy(model)
    args = _make_args(cfg)  # alpha 사용
    data_loaders = _pack_loaders(retain_loader=retain_loader, forget_loader=forget_loader)
    crit, opt = _criterion_optimizer(m, cfg)

    for epoch in range(args.unlearn_epochs):
        RUM_NG(data_loaders, m, crit, opt, epoch, args, mask=mask)
    return m

def unlearn_scrub_rum(model, retain_loader, forget_loader, cfg):
    m = copy.deepcopy(model)
    args = _make_args(cfg)  # beta, gamma, msteps, kd_T 사용
    data_loaders = _pack_loaders(retain_loader=retain_loader, forget_loader=forget_loader)
    crit, opt = _criterion_optimizer(m, cfg)

    for epoch in range(args.unlearn_epochs):
        RUM_SCRUB(data_loaders, m, crit, opt, epoch, args)
    return m

def unlearn_wfisher_rum(model, retain_loader, forget_loader, cfg):
    # 주의: Wfisher는 외부 의존(woodfisher/역헤시안 근사)이 있을 수 있음. 없으면 ImportError 발생.
    m = copy.deepcopy(model)
    args = _make_args(cfg)  # alpha 사용
    data_loaders = _pack_loaders(retain_loader=retain_loader, forget_loader=forget_loader)
    crit, opt = _criterion_optimizer(m, cfg)

    for epoch in range(args.unlearn_epochs):
        RUM_WFISHER(data_loaders, m, crit, opt, epoch, args)
    return m

def unlearn_salun_rum(model, retain_loader, forget_loader, cfg, mask=None):
    # README 기준 RL = SalUn. saliency mask가 있으면 cfg["saliency_mask_path"]에 경로 저장 후 args로 넘기세요.
    m = copy.deepcopy(model)
    args = _make_args(cfg, path=cfg.get("saliency_mask_path", None))
    data_loaders = _pack_loaders(retain_loader=retain_loader, forget_loader=forget_loader)
    crit, opt = _criterion_optimizer(m, cfg)

    for epoch in range(args.unlearn_epochs):
        RUM_SALUN(data_loaders, m, crit, opt, epoch, args, mask=mask)
    return m

def unlearn_random_label_rum(model, retain_loader, forget_loader, cfg):
    m = copy.deepcopy(model)
    args = _make_args(cfg)
    data_loaders = _pack_loaders(retain_loader=retain_loader, forget_loader=forget_loader)
    crit, opt = _criterion_optimizer(m, cfg)

    for epoch in range(args.unlearn_epochs):
        RUM_RANDOM_LABEL(data_loaders, m, crit, opt, epoch, args)
    return m
