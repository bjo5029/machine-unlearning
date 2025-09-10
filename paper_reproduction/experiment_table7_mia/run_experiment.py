import os, copy, numpy as np, pandas as pd
from scipy import stats
import torch

from config import CONFIG
from data_es import load_cifar10_with_train_eval, split_retain_forget, create_es_partitions, create_es_partitions_balanced, create_loss_partitions, create_es_partitions_paper
from model_train import get_model, train_model, evaluate_model
from methods import (
    unlearn_finetune, unlearn_l1_sparse, unlearn_neggrad,
    unlearn_neggrad_plus, unlearn_salun, unlearn_random_label
)
from mia import calculate_mia_score  
from diagnostics import print_split_stats, _print_partition_set_es


def ci95_str(xs):
    """결과 리스트(xs)의 평균과 95% 신뢰구간을 문자열로 포맷팅"""
    xs = np.array(xs); mu = xs.mean() if len(xs) else 0.0
    if len(xs) < 2: return f"{mu:.3f}"
    sem = stats.sem(xs); ci = sem * stats.t.ppf(0.975, len(xs)-1)
    return f"{mu:.3f} ± {ci:.3f}"

def main():
    os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

    # 1) 데이터셋 분리: train(augment O), train_eval(augment X), test(augment X)
    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval("./data")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # 결과 집계: 각 방법 × ES 레벨 → F/R/T/M 리스트
    methods = ["Retrain","Original","Fine-tune","L1-sparse","NegGrad","NegGrad+","SalUn","Random-label"]
    es_levels = ["Low ES","Medium ES","High ES"]
    res = {m:{es:{"F":[], "R":[], "T":[], "M":[]} for es in es_levels} for m in methods}

    for run in range(CONFIG["num_runs"]):
        print(f"\n================ Run {run+1}/{CONFIG['num_runs']} ================")

        # 원 모델
        orig = get_model(CONFIG["device"])
        orig_pth = os.path.join(CONFIG["model_save_dir"], f"run_{run}_original_model.pth")
        part_pth = os.path.join(CONFIG["model_save_dir"], f"run_{run}_es_partitions.pth")

        # 2) 원모델 학습/로드 + ES 파티션(평가용 변환으로 생성)
        if CONFIG["run_training"] or (not os.path.exists(orig_pth) or not os.path.exists(part_pth)):
            print("\n[TRAIN] Step 1/5: train original")
            tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
            train_model(orig, tr_loader, CONFIG["epochs"], CONFIG["lr"], CONFIG["device"], CONFIG["momentum"], CONFIG["weight_decay"])
            torch.save(orig.state_dict(), orig_pth)

            # parts = create_es_partitions(orig, train_eval_ds, CONFIG["device"], CONFIG["batch_size"], CONFIG["forget_set_size"])
            # # 변경:
            # parts = create_es_partitions_balanced(
            #     original_model=orig,
            #     dataset_for_es=train_eval_ds,
            #     device=CONFIG["device"],
            #     batch_size=CONFIG["batch_size"],
            #     forget_set_size=CONFIG["forget_set_size"],
            #     bins=5,  # 3~10 권장
            # )
            # parts = create_loss_partitions(
            #     model=orig,
            #     dataset_eval=train_eval_ds,         # augment X 평가 변환
            #     device=CONFIG["device"],
            #     forget_set_size=CONFIG["forget_set_size"],
            #     num_classes=10,
            #     batch_size=CONFIG["batch_size"]
            # )
            parts = create_es_partitions_paper(
                original_model=orig,
                dataset_for_es=train_eval_ds,      # augment X
                device=CONFIG["device"],
                batch_size=CONFIG["batch_size"],
                forget_set_size=CONFIG["forget_set_size"]
            )
            torch.save(parts, part_pth)
            _print_partition_set_es(orig, train_eval_ds, parts, CONFIG["device"], CONFIG["batch_size"])
        else:
            print("\n[LOAD] original & ES partitions")
            orig.load_state_dict(torch.load(orig_pth, map_location=CONFIG["device"]))
            parts = torch.load(part_pth)
            _print_partition_set_es(orig, train_eval_ds, parts, CONFIG["device"], CONFIG["batch_size"])

        # 3) ES 레벨별 실험
        for es, forget_idx in parts.items():
            print(f"\n--- ES Level: {es} ---")

            # retain/forget 인덱스
            retain_idx, forget_idx = split_retain_forget(len(train_ds), forget_idx)

            # 학습용(augment O)
            retain_train_set = torch.utils.data.Subset(train_ds, retain_idx)
            forget_train_set = torch.utils.data.Subset(train_ds, forget_idx)
            retain_train_loader = torch.utils.data.DataLoader(retain_train_set, batch_size=CONFIG["batch_size"], shuffle=True)
            forget_train_loader = torch.utils.data.DataLoader(forget_train_set, batch_size=CONFIG["batch_size"], shuffle=True)

            # 평가용(augment X)
            retain_eval_set = torch.utils.data.Subset(train_eval_ds, retain_idx)
            forget_eval_set = torch.utils.data.Subset(train_eval_ds, forget_idx)
            retain_eval_loader = torch.utils.data.DataLoader(retain_eval_set, batch_size=CONFIG["batch_size"], shuffle=False)
            forget_eval_loader = torch.utils.data.DataLoader(forget_eval_set, batch_size=CONFIG["batch_size"], shuffle=False)

            # 진단: 원본 모델과 재학습 모델의 Retain/Forget 성능 확인
            print_split_stats(orig, retain_eval_loader,  CONFIG["device"], f"{es} / retain_eval (orig)")
            print_split_stats(orig, forget_eval_loader,  CONFIG["device"], f"{es} / forget_eval (orig)")

            # Retrain 모델 학습 및 평가 (언러닝의 이상적인 상한선)
            retr = get_model(CONFIG["device"])
            retr_pth = os.path.join(CONFIG["model_save_dir"], f"run_{run}_{es.replace(' ','')}_retrained.pth")
            if CONFIG["run_training"] or not os.path.exists(retr_pth):
                print("[TRAIN] Step 2/5: retrain on retain (augment O)")
                train_model(retr, retain_train_loader, CONFIG["epochs"], CONFIG["lr"], CONFIG["device"], CONFIG["momentum"], CONFIG["weight_decay"])
                torch.save(retr.state_dict(), retr_pth)
            else:
                retr.load_state_dict(torch.load(retr_pth, map_location=CONFIG["device"]))

            # retr 로더 만든 뒤/로드한 뒤
            print_split_stats(retr, retain_eval_loader, CONFIG["device"], f"{es} / retain_eval (retrain)")
            print_split_stats(retr, forget_eval_loader, CONFIG["device"], f"{es} / forget_eval (retrain)")

            print("\n[Step 3/5] Eval retrained (augment X for eval)...")
            rF = evaluate_model(retr, forget_eval_loader, CONFIG["device"])
            rR = evaluate_model(retr, retain_eval_loader, CONFIG["device"])
            rT = evaluate_model(retr, test_loader,            CONFIG["device"])
            rM = calculate_mia_score(retr, retain_train_loader, retain_eval_loader, forget_eval_loader, test_loader, CONFIG["device"])
            print(f"  Retrain -> F:{rF:.2f} R:{rR:.2f} T:{rT:.2f} M:{rM:.3f}")
            res["Retrain"][es]["F"].append(rF); res["Retrain"][es]["R"].append(rR)
            res["Retrain"][es]["T"].append(rT); res["Retrain"][es]["M"].append(rM)

            # 4) 모든 언러닝 기법 적용 및 평가
            def eval_and_log(name, model):
                """주어진 모델을 평가하고 결과를 res 딕셔너리에 기록"""
                uF = evaluate_model(model, forget_eval_loader, CONFIG["device"])
                uR = evaluate_model(model, retain_eval_loader, CONFIG["device"])
                uT = evaluate_model(model, test_loader,            CONFIG["device"])
                uM = calculate_mia_score(model, retain_train_loader, retain_eval_loader, forget_eval_loader, test_loader, CONFIG["device"])
                print(f"    - {name:<12s} F:{uF:.2f} R:{uR:.2f} T:{uT:.2f} M:{uM:.3f}")
                res[name][es]["F"].append(uF); res[name][es]["R"].append(uR)
                res[name][es]["T"].append(uT); res[name][es]["M"].append(uM)

            print("\n[Step 4/5] Apply & eval unlearning...")
            # Original
            eval_and_log("Original", copy.deepcopy(orig))
            # Fine-tune / L1 / NegGrad / NegGrad+ / SalUn / Random-label
            eval_and_log("Fine-tune",    unlearn_finetune(copy.deepcopy(orig), retain_train_loader, CONFIG))
            eval_and_log("L1-sparse",    unlearn_l1_sparse(copy.deepcopy(orig), retain_train_loader, CONFIG))
            eval_and_log("NegGrad",      unlearn_neggrad(copy.deepcopy(orig),   forget_train_loader, CONFIG))
            eval_and_log("NegGrad+",     unlearn_neggrad_plus(copy.deepcopy(orig), retain_train_loader, forget_train_loader, CONFIG))
            eval_and_log("SalUn",        unlearn_salun(copy.deepcopy(orig),     forget_train_set, CONFIG))
            eval_and_log("Random-label", unlearn_random_label(copy.deepcopy(orig), forget_train_set, CONFIG))

    # 5) 요약 표 (MIA 전용)
    print("\n================ Final Results (MIA) ================")
    for es in es_levels:
        print(f"\n--- {es} ---")
        rows = []
        for m in methods:
            rows.append({
                "Method":     m,
                "Forget Acc": ci95_str(res[m][es]["F"]),
                "Retain Acc": ci95_str(res[m][es]["R"]),
                "Test Acc":   ci95_str(res[m][es]["T"]),
                "MIA":        ci95_str(res[m][es]["M"]),
            })
        print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
