import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.paths import OUTPUT_DIR


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for candidate in candidates:
        matches = [column for column in df.columns if candidate in column]
        if matches:
            return matches[0]
    raise KeyError(f"None of these columns were found: {candidates}")


def objective_columns(df: pd.DataFrame) -> tuple[str, str]:
    co2_col = find_column(df, ["grid_co2_emission", "Grid CO2 Emission (ton/yr)"])
    cash_col = find_column(df, ["annual_net_cash_flow", "Total annual net cash flow"])
    return co2_col, cash_col


def denormalized_objectives(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    co2_col, cash_col = objective_columns(df)
    co2 = df[co2_col].astype(float)
    cash = df[cash_col].astype(float)
    if co2_col == "grid_co2_emission":
        co2 = co2 * 1e4
    if cash_col == "annual_net_cash_flow":
        cash = cash * 1e6
    return co2, cash


def minmax_normalize_topsis(co2: pd.Series, cash: pd.Series) -> pd.DataFrame:
    obj = pd.DataFrame({"co2": co2, "cash": cash})
    finite_mask = np.isfinite(obj["co2"]) & np.isfinite(obj["cash"])
    obj = obj.loc[finite_mask].copy()
    if obj.empty:
        raise ValueError("No finite CO2/cash objective rows available for TOPSIS.")

    co2_min, co2_max = obj["co2"].min(), obj["co2"].max()
    cash_min, cash_max = obj["cash"].min(), obj["cash"].max()
    co2_range = co2_max - co2_min
    cash_range = cash_max - cash_min

    obj["co2_norm"] = 1.0 if co2_range == 0 else (co2_max - obj["co2"]) / co2_range
    obj["cash_norm"] = 1.0 if cash_range == 0 else (obj["cash"] - cash_min) / cash_range

    norm_vals = obj[["co2_norm", "cash_norm"]].to_numpy(dtype=float)
    ideal = np.array([1.0, 1.0])
    negative = np.array([0.0, 0.0])
    obj["d_to_ideal"] = np.linalg.norm(norm_vals - ideal, axis=1)
    obj["d_to_negative"] = np.linalg.norm(norm_vals - negative, axis=1)
    obj["topsis_cc"] = obj["d_to_negative"] / (obj["d_to_ideal"] + obj["d_to_negative"])
    obj["topsis_rank"] = obj["topsis_cc"].rank(method="min", ascending=False).astype(int)
    return obj


def default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_topsis.csv")


def extract_topsis(input_csv: Path, output_csv: Optional[Path] = None) -> Path:
    df = pd.read_csv(input_csv)
    co2, cash = denormalized_objectives(df)
    topsis = minmax_normalize_topsis(co2, cash)
    best_index = int(topsis["topsis_cc"].idxmax())

    output_row = df.loc[[best_index]].copy()
    extra_cols = [
        "co2",
        "cash",
        "co2_norm",
        "cash_norm",
        "d_to_ideal",
        "d_to_negative",
        "topsis_cc",
        "topsis_rank",
    ]
    for col in extra_cols:
        output_row[col] = topsis.loc[best_index, col]
    output_row["source_csv"] = str(input_csv)
    output_row["source_index"] = best_index

    output_path = default_output_path(input_csv) if output_csv is None else output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_row.to_csv(output_path, index=False)

    print(f"Best solution is at index  {best_index}")
    print(f"Closeness coefficient     = {topsis.loc[best_index, 'topsis_cc']:.4f}")
    print(f"Saved TOPSIS point to     = {output_path}")
    return output_path


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    output_path = OUTPUT_DIR / path
    if output_path.exists():
        return output_path
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", nargs="+", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.output_csv is not None and len(args.input_csv) > 1:
        parser.error("--output-csv can only be used with one --input-csv.")

    for input_csv_text in args.input_csv:
        input_csv = resolve_input_path(input_csv_text)
        output_csv = Path(args.output_csv) if args.output_csv is not None else None
        extract_topsis(input_csv, output_csv)


if __name__ == "__main__":
    main()
