# summarize_csv.py (최종 수정본)

from pathlib import Path
import pandas as pd
import re
import argparse
from typing import Optional

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None

def fmt(x, precision=4):
    try:
        return f"{float(x):.{precision}f}"
    except (ValueError, TypeError):
        return "NA"

def process_file(path: Path) -> list[str]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [f"=== {path.stem} ===", f"[WARN] read failed: {e}", ""]

    df = normalize_columns(df)

    col_method = pick_col(df, ["method"])
    col_stage  = pick_col(df, ["stage"])
    col_retF   = pick_col(df, ["Retain_F"])
    col_retR   = pick_col(df, ["Retrain_R"])
    col_retT   = pick_col(df, ["Retrain_T"])
    col_uF     = pick_col(df, ["Unlearn_F"])
    col_uR     = pick_col(df, ["Unlearn_R"])
    col_uT     = pick_col(df, ["Unlearn_T"])
    col_dF     = pick_col(df, ["ΔF"])
    col_dR     = pick_col(df, ["ΔR"])
    col_dT     = pick_col(df, ["ΔT"])
    col_mia    = pick_col(df, ["MIA"])
    col_pdiff  = pick_col(df, ["PredDiff(%)"])

    if df.empty:
        return [f"=== {path.stem} ===", "(no rows after filters)", ""]

    num_cols = [col_retF, col_retR, col_retT, col_uF, col_uR, col_uT, col_dF, col_dR, col_dT, col_mia, col_pdiff]
    for c in num_cols:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    lines = [f"=== {path.stem} ==="]
    rows = []

    rF_val, rR_val, rT_val = None, None, None
    if not df.empty:
        first_row = df.iloc[0]
        # 벤치마크 값 저장
        if col_retF and pd.notna(first_row[col_retF]): rF_val = first_row[col_retF]
        if col_retR and pd.notna(first_row[col_retR]): rR_val = first_row[col_retR]
        if col_retT and pd.notna(first_row[col_retT]): rT_val = first_row[col_retT]
        
        rows.append((
            "Retrain (Benchmark)",
            fmt(rF_val, 2), fmt(rR_val, 2), fmt(rT_val, 2),
            "----", "----", "----"
        ))

    for _, row in df.iterrows():
        method_name = str(row[col_method]) if col_method else "NA"
        stage_str = str(row[col_stage]) if col_stage and pd.notna(row[col_stage]) else ""
        
        epoch_info = ""
        match = re.search(r'\(best_ep(\d+)\)', stage_str)
        if match:
            epoch_num = match.group(1)
            epoch_info = f" (ep{epoch_num})"
        
        formatted_method_name = f"{method_name}{epoch_info}"
        
        def format_with_delta(val_col, delta_col, is_abs_delta=False):
            val_str = fmt(row[val_col], 2) if val_col else "NA"
            if val_col and delta_col and pd.notna(row[val_col]) and pd.notna(row[delta_col]):
                delta_val = row[delta_col]
                if is_abs_delta:
                    val_str = f"{row[val_col]:.2f} (-{delta_val:.2f})"
                else:
                    val_str = f"{row[val_col]:.2f} ({delta_val:+.2f})"
            return val_str

        forget_str = format_with_delta(col_uF, col_dF, is_abs_delta=False)
        retain_str = format_with_delta(col_uR, col_dR, is_abs_delta=True)
        test_str = format_with_delta(col_uT, col_dT, is_abs_delta=True)
        
        # ▼▼▼▼▼ [추가] TotalAccDiff 계산 로직 ▼▼▼▼▼
        total_acc_diff_str = "NA"
        if all(c and pd.notna(row[c]) for c in [col_uF, col_uR, col_uT]) and all(v is not None for v in [rF_val, rR_val, rT_val]):
            uF_val, uR_val, uT_val = row[col_uF], row[col_uR], row[col_uT]
            total_acc_diff = abs(uF_val - rF_val) + abs(uR_val - rR_val) + abs(uT_val - rT_val)
            total_acc_diff_str = fmt(total_acc_diff, 2)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        rows.append((
            formatted_method_name,
            forget_str,
            retain_str,
            test_str,
            total_acc_diff_str, # 새로 추가된 값
            fmt(row[col_mia], 4) if col_mia else "NA",
            f"{row[col_pdiff]:.2f}%" if col_pdiff and pd.notna(row[col_pdiff]) else "NA"
        ))

    # ▼▼▼▼▼ [수정] 컬럼 이름에 TotalAccDiff 추가 ▼▼▼▼▼
    col_names = ["Method", "ForgetAcc", "RetainAcc", "TestAcc", "TotalAccDiff", "MIA", "PredDiff"]
    if not rows:
        return lines + ["(No data to display)",""]
        
    widths = [max(len(str(r[i])) for r in rows + [tuple(col_names)]) for i in range(len(col_names))]
    header = " | ".join(col_names[i].center(widths[i]) for i in range(len(col_names)))
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        line_items = []
        for i, item in enumerate(r):
            if i == 0:
                line_items.append(str(item).ljust(widths[i]))
            else:
                line_items.append(str(item).rjust(widths[i]))
        lines.append(" | ".join(line_items))
    lines.append("")
    return lines

def save_group(root: Path, keyword: str, part_index: int, out_file: str):
    paths = []
    for p in root.rglob("results_*.csv"):
        parts = p.stem.lower().split('_')
        if len(parts) > part_index and parts[part_index] == keyword:
            paths.append(p)
    
    paths = sorted(paths)
    lines_all: list[str] = []
    for p in paths:
        lines_all.extend(process_file(p))
    
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text("\n".join(lines_all), encoding="utf-8")
    print(f"Saved -> {Path(out_file).resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="saved_models", help="CSV 루트 디렉토리")
    ap.add_argument("--out_stage",  default="saved_models/summary/merged_stage.txt")
    ap.add_argument("--out_batch",  default="saved_models/summary/merged_batch.txt")
    ap.add_argument("--out_sample", default="saved_models/summary/merged_sample.txt")
    ap.add_argument("--out_random", default="saved_models/summary/merged_random.txt")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: Root directory not found at '{root.resolve()}'")
        return
    
    save_group(root, "stage",  3, args.out_stage)
    save_group(root, "batch",  3, args.out_batch)
    save_group(root, "sample", 3, args.out_sample)
    save_group(root, "random", 2, args.out_random)

if __name__ == "__main__":
    main()
    