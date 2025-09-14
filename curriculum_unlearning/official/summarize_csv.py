# merge_results_split_fixed.py
# saved_models 아래 results_*.csv를 읽어
# - stage, batch, sample 각각 별도 텍스트(merged_stage.txt, merged_batch.txt, merged_sample.txt)로 저장
# - stage 파일만 stage==3 필터 적용
# - method에 'wfisher' 포함(대소문자 무관) 제거
# - 표 형태 간격 정렬 (method, forget, retain, test, mia)
# - 섹션 첫 줄 'retrain' 행 값은 다음 컬럼에서 채움:
#     forget <- Retain_F
#     retain <- Retrain_R
#     test   <- Retrain_T
#   (요청 사항에 따라 기존 fine-tune/Unlearn_*가 아닌 위 3개 컬럼을 사용)

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

def fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "NA"

def process_file(path: Path, is_stage: bool) -> list[str]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [f"=== {path.stem} ===", f"[WARN] read failed: {e}", ""]

    df = normalize_columns(df)

    # 필요한 컬럼 매핑
    col_method = pick_col(df, ["method"])
    col_stage  = pick_col(df, ["stage"])
    # retrain 행에 사용할 3개 컬럼
    col_retF   = pick_col(df, ["Retain_F","retain_f","Retain F"])
    col_retR   = pick_col(df, ["Retrain_R","retrain_r","Retrain R"])
    col_retT   = pick_col(df, ["Retrain_T","retrain_t","Retrain T"])
    # 일반 표시에 사용할 unlearn F/R/T (없어도 동작)
    col_uF     = pick_col(df, ["Unlearn_F","unlearn_f","Unlearn f"])
    col_uR     = pick_col(df, ["Unlearn_R","unlearn_r","Unlearn r"])
    col_uT     = pick_col(df, ["Unlearn_T","unlearn_t","Unlearn t"])
    col_mia    = pick_col(df, ["MIA","mia"])

    # stage 파일만 stage==3 필터
    if is_stage and col_stage:
        df[col_stage] = pd.to_numeric(df[col_stage], errors="coerce")
        df = df[df[col_stage] == 3]

    # wfisher 제거
    if col_method:
        df = df[~df[col_method].astype(str).str.contains("wfisher", case=False, na=False)]

    if df.empty:
        return [f"=== {path.stem} ===", "(no rows after filters)", ""]

    # 숫자 변환
    for c in [col_retF, col_retR, col_retT, col_uF, col_uR, col_uT, col_mia]:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    lines = [f"=== {path.stem} ==="]

    rows = []

    # retrain 첫 줄(요청한 3개 컬럼 사용)
    first = df.iloc[0]
    rows.append((
        "retrain",
        fmt(first[col_retF]) if col_retF else "NA",  # forget  <- Retain_F
        fmt(first[col_retR]) if col_retR else "NA",  # retain  <- Retrain_R
        fmt(first[col_retT]) if col_retT else "NA",  # test    <- Retrain_T
        fmt(first[col_mia])  if col_mia  else "NA"
    ))

    # 나머지 행들: 기본적으로 Unlearn_F/R/T를 표에 사용(없으면 NA)
    for _, row in df.iterrows():
        m = str(row[col_method]) if col_method else "(method)"
        f = fmt(row[col_uF]) if col_uF else "NA"
        r = fmt(row[col_uR]) if col_uR else "NA"
        t = fmt(row[col_uT]) if col_uT else "NA"
        mia = fmt(row[col_mia]) if col_mia else "NA"
        rows.append((m, f, r, t, mia))

    # 열 너비 계산 및 표 렌더링
    col_names = ["method","forget","retain","test","mia"]
    widths = [max(len(str(r[i])) for r in rows + [tuple(col_names)]) for i in range(len(col_names))]
    header = " | ".join(col_names[i].ljust(widths[i]) for i in range(len(col_names)))
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))))
    lines.append("")
    return lines

def save_group(root: Path, keyword: str, out_file: str):
    paths = sorted(p for p in root.rglob("results_*.csv") if keyword in p.stem.lower())
    is_stage = (keyword == "stage")
    lines_all: list[str] = []
    for p in paths:
        lines_all.extend(process_file(p, is_stage=is_stage))
    Path(out_file).write_text("\n".join(lines_all), encoding="utf-8")
    print(f"Saved -> {Path(out_file).resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="saved_models", help="CSV 루트 디렉토리")
    ap.add_argument("--out_stage",  default="merged_stage.txt")
    ap.add_argument("--out_batch",  default="merged_batch.txt")
    ap.add_argument("--out_sample", default="merged_sample.txt")
    args = ap.parse_args()

    root = Path(args.root)
    save_group(root, "stage",  args.out_stage)
    save_group(root, "batch",  args.out_batch)
    save_group(root, "sample", args.out_sample)

if __name__ == "__main__":
    main()
