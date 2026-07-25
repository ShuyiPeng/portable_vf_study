import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.paths import OUTPUT_DIR


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (7, 6),
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
})


DESIGN_VARIABLES = [
    ("num_pv", "PV panels"),
    ("num_batt", "Batteries"),
    ("num_wind", "Wind turbines"),
    ("num_light", "LED lights"),
]

SERIES_COLORS = ["#1F78B4", "#E31A1C", "#33A02C", "#6A3D9A", "#FF7F00"]


def topsis_label(label: str) -> str:
    return f"{label.removesuffix(' solutions')} TOPSIS"


def resolve_csv_path(file_name: str, output_dir: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute() or path.exists():
        return path
    return output_dir / path


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for candidate in candidates:
        matches = [column for column in df.columns if candidate in column]
        if matches:
            return matches[0]
    raise KeyError(f"None of these columns were found: {candidates}")


def design_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    columns = []
    for column, label in DESIGN_VARIABLES:
        if column == "num_light":
            columns.append((find_column(df, ["num_light", "num_light_index"]), label))
        else:
            columns.append((find_column(df, [column]), label))
    return columns


def emission_column(df: pd.DataFrame) -> str:
    return find_column(df, ["Grid CO2 Emission (ton/yr)", "grid_co2_emission", "co2"])


def emission_values(df: pd.DataFrame) -> np.ndarray:
    column = emission_column(df)
    emissions = df[column].to_numpy(dtype=float)
    if column == "grid_co2_emission":
        emissions = emissions * 1e4
    return emissions


def cash_column(df: pd.DataFrame) -> str:
    return find_column(df, ["Total annual net cash flow", "annual_net_cash_flow", "cash"])


def cash_values(df: pd.DataFrame) -> np.ndarray:
    column = cash_column(df)
    cash = df[column].to_numpy(dtype=float)
    if column == "annual_net_cash_flow":
        cash = cash * 1e6
    return cash / 1e6


def topsis_optimal_index(emissions: np.ndarray, cash_million: np.ndarray) -> int:
    finite_mask = np.isfinite(emissions) & np.isfinite(cash_million)
    if not finite_mask.any():
        raise ValueError("No finite emission/cash objective rows available for TOPSIS.")

    finite_indices = np.flatnonzero(finite_mask)
    emissions_valid = emissions[finite_mask]
    cash_valid = cash_million[finite_mask]
    emissions_range = np.ptp(emissions_valid)
    cash_range = np.ptp(cash_valid)
    emissions_norm = (
        np.zeros_like(emissions_valid)
        if emissions_range == 0
        else (emissions_valid - emissions_valid.min()) / emissions_range
    )
    cash_norm = np.zeros_like(cash_valid) if cash_range == 0 else (cash_valid - cash_valid.min()) / cash_range
    distance_to_ideal = np.sqrt(emissions_norm**2 + (1 - cash_norm)**2)
    distance_to_nadir = np.sqrt((1 - emissions_norm)**2 + cash_norm**2)
    score = distance_to_nadir / (distance_to_ideal + distance_to_nadir)
    return int(finite_indices[np.nanargmax(score)])


def default_topsis_path(csv_file: Path) -> Path:
    return csv_file.with_name(f"{csv_file.stem}_topsis.csv")


def load_topsis_point(csv_file: Path, columns: list[tuple[str, str]]) -> Optional[dict[str, float]]:
    topsis_file = default_topsis_path(csv_file)
    if topsis_file.exists():
        topsis_df = pd.read_csv(topsis_file)
        if topsis_df.empty:
            return None

        topsis_point = {
            column: float(topsis_df[column].iloc[0])
            for column, _ in columns
            if column in topsis_df.columns
        }
        topsis_point["emission"] = float(emission_values(topsis_df)[0])
        return topsis_point

    df = pd.read_csv(csv_file)
    emissions = emission_values(df)
    cash_million = cash_values(df)
    optimal_index = topsis_optimal_index(emissions, cash_million)
    topsis_point = {column: float(df[column].iloc[optimal_index]) for column, _ in columns}
    topsis_point["emission"] = float(emissions[optimal_index])
    return topsis_point


def topsis_row_index(df: pd.DataFrame, topsis_point: Optional[dict[str, float]]) -> Optional[int]:
    if not topsis_point:
        return None

    mask = pd.Series(True, index=df.index)
    for column, value in topsis_point.items():
        if column == "emission":
            continue
        mask &= np.isclose(df[column].astype(float), value, equal_nan=True)

    matches = df.index[mask]
    if len(matches) == 0:
        return None
    return int(matches[0])


def style_axes(axes_flat):
    for ax in axes_flat:
        ax.grid(False)
        ax.tick_params(direction="out", which="both")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    for ax in axes_flat[-2:]:
        ax.set_xlabel("Grid CO2 Emissions (ton/year)")


def save_figure(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def plot_design_variables(csv_file: Path, output_dir: Path, output_stem: Optional[str] = None):
    df = pd.read_csv(csv_file)
    columns = design_columns(df)
    x = emission_values(df)

    topsis_point = load_topsis_point(csv_file, columns)
    topsis_index = topsis_row_index(df, topsis_point)

    fig, axes = plt.subplots(2, 2, sharex=True)
    axes_flat = axes.ravel()

    for ax, (column, label) in zip(axes_flat, columns):
        y = df[column].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            s=18,
            marker="o",
            edgecolors="#1F78B4",
            facecolors="none",
            linewidths=0.8,
            alpha=0.65,
        )

        if topsis_point is not None and column in topsis_point:
            ax.scatter(
                topsis_point["emission"],
                topsis_point[column],
                s=130,
                marker="*",
                color="#E31A1C",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="TOPSIS-optimal",
            )

        ax.set_ylabel(label)

    style_axes(axes_flat)

    if topsis_index is not None:
        axes_flat[0].legend(frameon=True, edgecolor="black", fancybox=False, loc="best")
        print(f"TOPSIS design point matched row index: {topsis_index}")
    elif topsis_point is not None:
        print("TOPSIS design point was loaded but did not match a row in the Pareto CSV.")

    stem = output_stem if output_stem is not None else f"{csv_file.stem}_design_variables"
    return save_figure(fig, output_dir, stem)


def plot_design_variable_comparison(
    csv_files: list[Path],
    labels: list[str],
    output_dir: Path,
    output_stem: Optional[str] = None,
):
    fig, axes = plt.subplots(2, 2, sharex=True)
    axes_flat = axes.ravel()

    for series_index, (csv_file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_file)
        columns = design_columns(df)
        x = emission_values(df)
        topsis_point = load_topsis_point(csv_file, columns)
        color = SERIES_COLORS[series_index % len(SERIES_COLORS)]

        for ax, (column, variable_label) in zip(axes_flat, columns):
            y = df[column].to_numpy(dtype=float)
            ax.scatter(
                x,
                y,
                s=18,
                marker="o",
                edgecolors=color,
                facecolors="none",
                linewidths=0.8,
                alpha=0.65,
                label=label if ax is axes_flat[0] else "_nolegend_",
            )

            if topsis_point is not None and column in topsis_point:
                ax.scatter(
                    topsis_point["emission"],
                    topsis_point[column],
                    s=130,
                    marker="*",
                    color=color,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=6,
                    label=topsis_label(label) if ax is axes_flat[0] else "_nolegend_",
                )

            ax.set_ylabel(variable_label)

    style_axes(axes_flat)
    axes_flat[0].legend(frameon=True, edgecolor="black", fancybox=False, loc="best")

    stem = output_stem if output_stem is not None else "design_variables_compare"
    return save_figure(fig, output_dir, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="One or more Pareto CSV files to plot.")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for the CSV files.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    csv_files = [resolve_csv_path(csv, output_dir) for csv in args.csv]
    labels = args.labels if args.labels is not None else [csv_file.stem for csv_file in csv_files]
    if len(labels) != len(csv_files):
        parser.error("--labels must have the same number of values as --csv.")

    if len(csv_files) == 1:
        pdf_path, png_path = plot_design_variables(csv_files[0], output_dir, args.output_stem)
    else:
        pdf_path, png_path = plot_design_variable_comparison(csv_files, labels, output_dir, args.output_stem)

    print(f"Saved design variables: {pdf_path}")
    print(f"Saved design variables: {png_path}")


if __name__ == "__main__":
    main()
