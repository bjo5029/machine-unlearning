import os, time, copy, numpy as np, pandas as pd
from scipy import stats
import torch

from config import CONFIG
from data_es import load_cifar10, split_retain_forget, create_es_partitions
from model_train import get_model, train_model, evaluate_model
from methods import (
    unlearn_finetune, unlearn_l1_sparse, unlearn_neggrad,
    unlearn_neggrad_plus, unlearn_salun, unlearn_random_label
)
from evaluation.tow import calculate_tow, calculate_prediction_diff


def ci95_str(xs):
    xs = np.array(xs); mu = xs.mean() if len(xs) else 0.0
    if len(xs) < 2: return f"{mu:.3f}"
    sem = stats.sem(xs); ci = sem * stats.t.ppf(0.975, len(xs)-1)
    return f"{mu:.3f} ± {ci:.3f}"

def main():
    os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
    train_ds, test_ds, full_train_eval_ds = load_cifar10("./data")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    full_train_eval_loader = torch.utils.data.DataLoader(full_train_eval_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    all_methods = ["Original","Fine-tune","L1-sparse","NegGrad","NegGrad+","SalUn","Random-label"]
    es_levels   = ["Low ES","Medium ES","High ES"]
    results = {
        "ToW":       {m:{es:[] for es in es_levels} for m in all_methods},
        "Pred_Diff": {m:{es:[] for es in es_levels} for m in all_methods},
    }

    for run in range(CONFIG["num_runs"]):
        print(f"\n================ Run {run+1}/{CONFIG['num_runs']} ================")
        orig = get_model(CONFIG["device"])
        orig_pth = os.path.join(CONFIG["model_save_dir"], f"run_{run}_original_model.pth")
        part_pth = os.path.join(CONFIG["model_save_dir"], f"run_{run}_es_partitions.pth")

        if CONFIG["run_training"]:
            print("\n[TRAIN] Step 1/5: train original")
            tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
            train_model(orig, tr_loader, CONFIG["epochs"], CONFIG["lr"], CONFIG["device"], CONFIG["momentum"], CONFIG["weight_decay"])
            torch.save(orig.state_dict(), orig_pth)
            parts = create_es_partitions(orig, train_ds, CONFIG["device"], CONFIG["batch_size"], CONFIG["forget_set_size"])
            torch.save(parts, part_pth)
        else:
            print("\n[LOAD] original & ES partitions")
            orig.load_state_dict(torch.load(orig_pth, map_location=CONFIG["device"]))
            parts = torch.load(part_pth)

        for es, forget_idx in parts.items():
            print(f"\n--- ES Level: {es} ---")
            retain_set, forget_set = split_retain_forget(train_ds, forget_idx)
            retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=CONFIG["batch_size"], shuffle=True)
            retain_eval   = torch.utils.data.DataLoader(retain_set, batch_size=CONFIG["batch_size"], shuffle=False)
            forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=CONFIG["batch_size"], shuffle=False)

            # retrained (retain only)
            retr = get_model(CONFIG["device"])
            retr_pth = os.path.join(CONFIG["model_save_dir"], f"run_{run}_{es.replace(' ','')}_retrained.pth")
            if CONFIG["run_training"]:
                print("[TRAIN] Step 2/5: retrain on retain")
                train_model(retr, retain_loader, CONFIG["epochs"], CONFIG["lr"], CONFIG["device"], CONFIG["momentum"], CONFIG["weight_decay"])
                torch.save(retr.state_dict(), retr_pth)
            else:
                retr.load_state_dict(torch.load(retr_pth, map_location=CONFIG["device"]))

            print("\n[Step 3/5] Eval retrained...")
            retr_accs = {
                "retain": evaluate_model(retr, retain_eval, CONFIG["device"]),
                "forget": evaluate_model(retr, forget_loader, CONFIG["device"]),
                "test":   evaluate_model(retr, test_loader,   CONFIG["device"]),
            }
            print(f"  Retrain Accs -> Retain:{retr_accs['retain']:.2f}%  Forget:{retr_accs['forget']:.2f}%  Test:{retr_accs['test']:.2f}%")

            methods = {
                "Original":      lambda: copy.deepcopy(orig),
                "Fine-tune":     lambda: unlearn_finetune(orig, retain_loader, CONFIG),
                "L1-sparse":     lambda: unlearn_l1_sparse(orig, retain_loader, CONFIG),
                "NegGrad":       lambda: unlearn_neggrad(orig, forget_loader, CONFIG),
                "NegGrad+":      lambda: unlearn_neggrad_plus(orig, retain_loader, forget_loader, CONFIG),
                "SalUn":         lambda: unlearn_salun(orig, forget_set, CONFIG),
                "Random-label":  lambda: unlearn_random_label(orig, forget_set, CONFIG),
            }

            print("\n[Step 4/5] Apply & eval unlearning...")
            for name, fn in methods.items():
                upth = os.path.join(CONFIG["model_save_dir"], f"run_{run}_{es.replace(' ','')}_{name}_unlearned.pth")
                if CONFIG["run_training"]:
                    print(f"  > train {name}")
                    u = fn(); torch.save(u.state_dict(), upth)
                else:
                    u = get_model(CONFIG["device"]); u.load_state_dict(torch.load(upth, map_location=CONFIG["device"]))

                u_accs = {
                    "retain": evaluate_model(u, retain_eval, CONFIG["device"]),
                    "forget": evaluate_model(u, forget_loader, CONFIG["device"]),
                    "test":   evaluate_model(u, test_loader,   CONFIG["device"]),
                }
                tow = calculate_tow(u_accs, retr_accs)
                pdiff = calculate_prediction_diff(u, retr, full_train_eval_loader, CONFIG["device"])
                results["ToW"][name][es].append(tow)
                results["Pred_Diff"][name][es].append(pdiff)

                print(f"    - {name:<12s} Retain:{u_accs['retain']:.2f}%  Forget:{u_accs['forget']:.2f}%  Test:{u_accs['test']:.2f}% | ToW:{tow:.3f}  PredDiff:{pdiff:.3f}%")

    # [Step 5/5] 요약 표
    print("\n================ Final Results ================")
    rows = []
    for m in all_methods:
        row = {"Method": m}
        for es in es_levels:
            row[f"ToW_{es}"] = ci95_str(results["ToW"][m][es])
        for es in es_levels:
            row[f"PredDiff_{es}"] = ci95_str(results["Pred_Diff"][m][es])
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "Method",
        "ToW_Low ES","ToW_Medium ES","ToW_High ES",
        "PredDiff_Low ES","PredDiff_Medium ES","PredDiff_High ES"
    ])
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
