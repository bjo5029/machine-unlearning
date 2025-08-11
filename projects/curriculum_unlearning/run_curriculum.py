import os, copy, numpy as np, pandas as pd, hashlib
from scipy import stats
import torch

from config import CONFIG
from seeds import set_seed
from data_es import (
    load_cifar10_with_train_eval, split_retain_forget, build_loaders,
    create_es_partitions_paper, create_es_partitions_balanced,
    hash_indices
)
from model_train import get_model, train_model, evaluate_model
from methods import unlearn_neggrad_plus  # 기본 비교용 (원하면 methods에서 교체)
from diagnostics import print_split_stats
from metrics import calculate_prediction_diff, calculate_mia_score

def _delta_metrics(rF, rR, rT, uF, uR, uT):
    dF = uF - rF
    dR = abs(uR - rR)
    dT = abs(uT - rT)
    return dF, dR, dT

def ci95(xs):
    xs = np.array(xs); mu = xs.mean() if len(xs) else 0.0
    if len(xs) < 2: return f"{mu:.3f}"
    sem = stats.sem(xs); tval = stats.t.ppf(0.975, len(xs)-1)
    return f"{mu:.3f} ± {(sem*tval):.3f}"

def make_stream(parts, order, seed=123):
    """parts: dict('High ES','Medium ES','Low ES' -> np.array idx)
       return: [S1, S2, S3]
    """
    if order == "HML":
        return [parts["High ES"], parts["Medium ES"], parts["Low ES"]]
    if order == "LMH":
        return [parts["Low ES"], parts["Medium ES"], parts["High ES"]]
    if order == "RND":
        allidx = np.concatenate([parts["High ES"], parts["Medium ES"], parts["Low ES"]])
        rng = np.random.default_rng(seed)
        rng.shuffle(allidx)
        third = len(allidx)//3
        return [allidx[:third], allidx[third:2*third], allidx[2*third:3*third]]
    raise ValueError("order must be one of HML/LMH/RND")

def retrain_cache_path(save_dir, retain_idx_sorted):
    h = hashlib.sha1(np.asarray(retain_idx_sorted, dtype=np.int64).tobytes()).hexdigest()
    return os.path.join(save_dir, f"retrain_{h}.pth")

def run_once(order, run_id, train_ds, train_eval_ds, test_loader, parts):
    device = CONFIG["device"]; bs = CONFIG["batch_size"]
    # 원모델 학습/로딩
    orig = get_model(device)
    orig_pth = os.path.join(CONFIG["model_save_dir"], f"run{run_id}_orig.pth")
    if CONFIG["run_training"] or not os.path.exists(orig_pth):
        tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
        print("[TRAIN] Original")
        train_model(orig, tr_loader, CONFIG["epochs"], CONFIG["lr"], device, CONFIG["momentum"], CONFIG["weight_decay"])
        torch.save(orig.state_dict(), orig_pth)
    else:
        orig.load_state_dict(torch.load(orig_pth, map_location=device))

    # 커리큘럼 스트림
    stream = make_stream(parts, order, seed=CONFIG["seed"]+run_id)
    method_name = "NegGrad+"

    # 누적 평가 테이블
    rows = []
    model_u = copy.deepcopy(orig)  # 언러닝 진행 모델
    F_used = np.array([], dtype=np.int64)

    # PD 계산용 풀 평가 로더
    full_train_eval_loader = torch.utils.data.DataLoader(train_eval_ds, batch_size=bs, shuffle=False)

    for stage, S in enumerate(stream, start=1):
        # 누적 forget & retain 인덱스
        F_used = np.unique(np.concatenate([F_used, S]))
        retain_idx, forget_idx = split_retain_forget(len(train_ds), F_used)

        # 로더 구성
        r_train, f_train, r_eval, f_eval = build_loaders(
            train_ds, train_eval_ds, retain_idx, forget_idx, bs, shuffle_train=True
        )

        # retrain_j (캐시)
        retr = get_model(device)
        retr_path = retrain_cache_path(CONFIG["model_save_dir"], np.sort(retain_idx))
        if CONFIG["run_training"] or not os.path.exists(retr_path):
            print(f"[TRAIN] Retrain stage{stage} on retain-only (|retain|={len(retain_idx)})")
            train_model(retr, r_train, CONFIG["epochs"], CONFIG["lr"], device, CONFIG["momentum"], CONFIG["weight_decay"])
            torch.save(retr.state_dict(), retr_path)
        else:
            retr.load_state_dict(torch.load(retr_path, map_location=device))

        # 언러닝 1 stage 적용
        print(f"[UNLEARN] {method_name} stage{stage} (|forget_added|={len(S)}, |forget_total|={len(F_used)})")
        model_u = unlearn_neggrad_plus(model_u, r_train, f_train, CONFIG)

        # 평가 (retrain vs unlearned)
        rF = evaluate_model(retr, f_eval, device)
        rR = evaluate_model(retr, r_eval, device)
        rT = evaluate_model(retr, test_loader, device)

        uF = evaluate_model(model_u, f_eval, device)
        uR = evaluate_model(model_u, r_eval, device)
        uT = evaluate_model(model_u, test_loader, device)

        # Δ지표
        dF, dR, dT = _delta_metrics(rF, rR, rT, uF, uR, uT)

        # 보조지표
        mia = calculate_mia_score(model_u, r_train, r_eval, f_eval, test_loader, device)
        pdiff = calculate_prediction_diff(model_u, retr, full_train_eval_loader, device)

        # 스테이지 콘솔 로그
        print(f"{order:>4} | stage {stage} | Ftot={len(F_used):>4d}  "
              f"Ret:{rF:5.2f}/{rR:5.2f}/{rT:5.2f}  "
              f"Unl:{uF:5.2f}/{uR:5.2f}/{uT:5.2f}  "
              f"ΔF:{dF:+6.2f}  ΔR:{dR:6.2f}  ΔT:{dT:6.2f}  "
              f"MIA:{mia:.6f}  PredDiff:{pdiff:.3f}")

        rows.append({
            "order": order, "stage": stage, "forget_total": len(F_used),
            "Retrain_F": rF, "Retrain_R": rR, "Retrain_T": rT,
            "Unlearn_F": uF, "Unlearn_R": uR, "Unlearn_T": uT,
            "ΔF": dF, "ΔR": dR, "ΔT": dT,
            "MIA": mia, "PredDiff(%)": pdiff
        })

    df = pd.DataFrame(rows)
    return df

def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

    # 데이터
    train_ds, train_eval_ds, test_ds = load_cifar10_with_train_eval(CONFIG["data_root"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # ES 파티션
    probe_model = get_model(CONFIG["device"])
    if CONFIG.get("use_paper_es", True):
        parts = create_es_partitions_paper(probe_model, train_eval_ds, CONFIG["device"], CONFIG["batch_size"], CONFIG["forget_set_size"])
    else:
        parts = create_es_partitions_balanced(
            probe_model, train_eval_ds, CONFIG["device"], CONFIG["batch_size"],
            CONFIG["forget_set_size"], bins=CONFIG.get("balanced_bins", 5)
        )

    # Run
    all_orders = ["HML","LMH","RND"]
    all_dfs = []
    for run in range(CONFIG["num_runs"]):
        for order in all_orders:
            print(f"\n===== Run {run+1}/{CONFIG['num_runs']} | Order={order} =====")
            df = run_once(order, run, train_ds, train_eval_ds, test_loader, parts)
            all_dfs.append(df)

    out = pd.concat(all_dfs, ignore_index=True)

    # 스테이지별 표 + 저장
    cols = ["order","stage","forget_total",
            "Retrain_F","Retrain_R","Retrain_T",
            "Unlearn_F","Unlearn_R","Unlearn_T",
            "ΔF","ΔR","ΔT",
            "MIA","PredDiff(%)"]
    out = out[cols]

    print("\n===== Stage-wise results =====")
    print(out.to_string(index=False))

    # 최종 스테이지만 요약
    last = out.sort_values(["order","stage"]).groupby("order").tail(1)
    print("\n===== Final (after last stage) =====")
    print(last[["order","Unlearn_F","Unlearn_R","Unlearn_T","ΔF","ΔR","ΔT","MIA","PredDiff(%)"]].to_string(index=False))

    # 저장
    csv_path = os.path.join(CONFIG["model_save_dir"], "curriculum_results.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

if __name__ == "__main__":
    main()
