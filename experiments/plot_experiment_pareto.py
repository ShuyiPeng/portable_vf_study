import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    "figure.figsize": (7, 4.5),
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
})


PLOT_CONFIGS = {
    "fixed_light": {
        "output_stem": "SP_fixed_light_pareto_compare",
        "title": None,
        "series": [
            {"label": "SP-S1", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Fixed-light 14812", "file": "SP_fixed_light_14812.csv", "marker": "^", "color": "#E31A1C"},
            {"label": "Fixed-light 18517", "file": "SP_fixed_light_18517.csv", "marker": "s", "color": "#33A02C"},
        ],
    },
    "price": {
        "output_stem": "SP_price_sensitivity_pareto",
        "title": None,
        "series": [
            {"label": "SP-S1", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Price 80%", "file": "SP_price_080.csv", "marker": "^", "color": "#33A02C"},
            {"label": "Price 120%", "file": "SP_price_120.csv", "marker": "s", "color": "#E31A1C"},
        ],
    },
    "export": {
        "output_stem": "SP_export_limit_pareto",
        "title": None,
        "series": [
            {"label": "SP-S1", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Export 10%", "file": "SP_export_010.csv", "marker": "^", "color": "#33A02C"},
            {"label": "Export 50%", "file": "SP_export_050.csv", "marker": "s", "color": "#E31A1C"},
        ],
    },
}

FIXED_LIGHT_MARKERS = ["^", "s", "D", "v", "P", "X"]
FIXED_LIGHT_COLORS = ["#E31A1C", "#33A02C", "#6A3D9A", "#FF7F00", "#A6CEE3", "#B15928"]


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
    co2_col = find_column(df, ["Grid CO2 Emission (ton/yr)", "grid_co2_emission"])
    cash_col = find_column(df, ["Total annual net cash flow", "annual_net_cash_flow"])
    return co2_col, cash_col


def objective_values(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    co2_col, cash_col = objective_columns(df)
    co2 = df[co2_col].to_numpy(dtype=float)
    cash = df[cash_col].to_numpy(dtype=float)
    if co2_col == "grid_co2_emission":
        co2 = co2 * 1e4
    if cash_col == "annual_net_cash_flow":
        cash = cash * 1e6
    return co2, cash / 1e6


def objective_point(df: pd.DataFrame) -> tuple[float, float]:
    co2, cash_million = objective_values(df)
    if len(co2) == 0:
        raise ValueError("TOPSIS CSV does not contain any rows.")
    return float(co2[0]), float(cash_million[0])


def topsis_optimal_index(co2: np.ndarray, cash_million: np.ndarray) -> int:
    finite_mask = np.isfinite(co2) & np.isfinite(cash_million)
    if not finite_mask.any():
        raise ValueError("No finite CO2/cash objective rows available for TOPSIS.")

    finite_indices = np.flatnonzero(finite_mask)
    co2_valid = co2[finite_mask]
    cash_valid = cash_million[finite_mask]
    co2_range = np.ptp(co2_valid)
    cash_range = np.ptp(cash_valid)
    co2_norm = np.zeros_like(co2_valid) if co2_range == 0 else (co2_valid - co2_valid.min()) / co2_range
    cash_norm = np.zeros_like(cash_valid) if cash_range == 0 else (cash_valid - cash_valid.min()) / cash_range
    distance_to_ideal = np.sqrt(co2_norm**2 + (1 - cash_norm)**2)
    distance_to_nadir = np.sqrt((1 - co2_norm)**2 + cash_norm**2)
    score = distance_to_nadir / (distance_to_ideal + distance_to_nadir)
    return int(finite_indices[np.nanargmax(score)])


def resolve_csv_path(file_name: str, output_dir: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute() or path.exists():
        return path
    return output_dir / path


def fixed_light_series(files: list[str], labels: Optional[list[str]] = None) -> list[dict]:
    labels = labels if labels is not None else [Path(file_name).stem for file_name in files]
    series = [{"label": "SP-S1", "file": None, "marker": "o", "color": "#1F78B4"}]
    for index, (file_name, label) in enumerate(zip(files, labels)):
        series.append({
            "label": label,
            "file": file_name,
            "marker": FIXED_LIGHT_MARKERS[index % len(FIXED_LIGHT_MARKERS)],
            "color": FIXED_LIGHT_COLORS[index % len(FIXED_LIGHT_COLORS)],
        })
    return series


def plot_config(
    case_name: str,
    fixed_light_files: Optional[list[str]] = None,
    fixed_light_labels: Optional[list[str]] = None,
    output_stem: Optional[str] = None,
) -> dict:
    config = dict(PLOT_CONFIGS[case_name])
    if case_name == "fixed_light" and fixed_light_files is not None:
        config["series"] = fixed_light_series(fixed_light_files, fixed_light_labels)
    if output_stem is not None:
        config["output_stem"] = output_stem
    return config


def default_topsis_path(csv_file: Path) -> Path:
    return csv_file.with_name(f"{csv_file.stem}_topsis.csv")


def load_topsis_point(csv_file: Path, co2: np.ndarray, cash_million: np.ndarray) -> tuple[float, float, str]:
    topsis_file = default_topsis_path(csv_file)
    if topsis_file.exists():
        topsis_df = pd.read_csv(topsis_file)
        co2_point, cash_point = objective_point(topsis_df)
        return co2_point, cash_point, str(topsis_file)

    optimal_index = topsis_optimal_index(co2, cash_million)
    return float(co2[optimal_index]), float(cash_million[optimal_index]), "computed"


def plot_case(
    case_name: str,
    baseline_csv: Path,
    output_dir: Path,
    fixed_light_files: Optional[list[str]] = None,
    fixed_light_labels: Optional[list[str]] = None,
    output_stem: Optional[str] = None,
):
    config = plot_config(case_name, fixed_light_files, fixed_light_labels, output_stem)
    fig, ax = plt.subplots()
    legend_elements = [Line2D([0], [0], color="w", label=r"$\mathbf{Scenarios}$")]

    for series in config["series"]:
        csv_file = baseline_csv if series["file"] is None else resolve_csv_path(series["file"], output_dir)
        df = pd.read_csv(csv_file)
        co2, cash_million = objective_values(df)

        ax.scatter(
            co2,
            cash_million,
            marker=series["marker"],
            s=18,
            edgecolors=series["color"],
            facecolors="none",
            linewidths=0.8,
            alpha=0.45,
            label="_nolegend_",
        )

        topsis_co2, topsis_cash_million, topsis_source = load_topsis_point(csv_file, co2, cash_million)
        ax.scatter(
            topsis_co2,
            topsis_cash_million,
            marker="*",
            s=130,
            color=series["color"],
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label="_nolegend_",
        )
        print(f"{case_name} / {series['label']} TOPSIS: {topsis_source}")

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=series["marker"],
                color="w",
                label=series["label"],
                markeredgecolor=series["color"],
                markerfacecolor="none",
                markersize=7,
            )
        )

    legend_elements.extend([
        Line2D([0], [0], color="w", label=""),
        Line2D([0], [0], color="w", label=r"$\mathbf{Notation}$"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Open markers: Pareto solutions",
            markeredgecolor="gray",
            markerfacecolor="none",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Stars: TOPSIS-optimal",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=11,
        ),
    ])

    if config["title"]:
        ax.set_title(config["title"])
    ax.set_xlabel("Grid CO₂ Emissions (ton/year)")
    ax.set_ylabel("Annual Net Profit (10⁶ €/year)")
    ax.grid(False)
    ax.tick_params(direction="out", which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        borderpad=0.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{config['output_stem']}.pdf"
    png_path = output_dir / f"{config['output_stem']}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--output-stem", default=None)
    parser.add_argument(
        "--fixed-light-files",
        nargs="+",
        default=None,
        help="Fixed-light CSV files to compare when --case fixed_light is used.",
    )
    parser.add_argument(
        "--fixed-light-labels",
        nargs="+",
        default=None,
        help="Legend labels for --fixed-light-files.",
    )
    parser.add_argument(
        "--case",
        choices=["all", *PLOT_CONFIGS.keys()],
        default="all",
    )
    args = parser.parse_args()

    baseline_csv = Path(args.baseline_csv)
    output_dir = Path(args.output_dir)
    if args.fixed_light_labels is not None and args.fixed_light_files is None:
        parser.error("--fixed-light-labels can only be used with --fixed-light-files.")
    if args.fixed_light_labels is not None and len(args.fixed_light_labels) != len(args.fixed_light_files):
        parser.error("--fixed-light-labels must have the same number of values as --fixed-light-files.")
    cases = PLOT_CONFIGS.keys() if args.case == "all" else [args.case]
    for case_name in cases:
        pdf_path, png_path = plot_case(
            case_name,
            baseline_csv,
            output_dir,
            fixed_light_files=args.fixed_light_files,
            fixed_light_labels=args.fixed_light_labels,
            output_stem=args.output_stem if args.case != "all" else None,
        )
        print(f"Saved {case_name}: {pdf_path}")
        print(f"Saved {case_name}: {png_path}")


if __name__ == "__main__":
    main()
