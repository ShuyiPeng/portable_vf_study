import argparse
import sys
from pathlib import Path

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
            {"label": "Baseline", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Fixed light", "file": "SP_fixed_light_14812.csv", "marker": "^", "color": "#E31A1C"},
        ],
    },
    "price": {
        "output_stem": "SP_price_sensitivity_pareto",
        "title": None,
        "series": [
            {"label": "Baseline", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Price 80%", "file": "SP_price_080.csv", "marker": "^", "color": "#33A02C"},
            {"label": "Price 120%", "file": "SP_price_120.csv", "marker": "s", "color": "#E31A1C"},
        ],
    },
    "export": {
        "output_stem": "SP_export_limit_pareto",
        "title": None,
        "series": [
            {"label": "Baseline", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Export 10%", "file": "SP_export_010.csv", "marker": "^", "color": "#33A02C"},
            {"label": "Export 50%", "file": "SP_export_050.csv", "marker": "s", "color": "#E31A1C"},
        ],
    },
}


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


def topsis_optimal_index(co2: np.ndarray, cash_million: np.ndarray) -> int:
    co2_range = np.ptp(co2)
    cash_range = np.ptp(cash_million)
    co2_norm = np.zeros_like(co2) if co2_range == 0 else (co2 - co2.min()) / co2_range
    cash_norm = np.zeros_like(cash_million) if cash_range == 0 else (cash_million - cash_million.min()) / cash_range
    distance_to_ideal = np.sqrt(co2_norm**2 + (1 - cash_norm)**2)
    distance_to_nadir = np.sqrt((1 - co2_norm)**2 + cash_norm**2)
    score = distance_to_nadir / (distance_to_ideal + distance_to_nadir)
    return int(np.nanargmax(score))


def resolve_csv_path(file_name: str, output_dir: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return output_dir / path


def plot_case(case_name: str, baseline_csv: Path, output_dir: Path):
    config = PLOT_CONFIGS[case_name]
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

        if series["file"] is not None:
            optimal_index = topsis_optimal_index(co2, cash_million)
            ax.scatter(
                co2[optimal_index],
                cash_million[optimal_index],
                marker="*",
                s=130,
                color=series["color"],
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="_nolegend_",
            )

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
            label="Pareto frontier",
            markeredgecolor="gray",
            markerfacecolor="none",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="TOPSIS-optimal",
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
    parser.add_argument(
        "--case",
        choices=["all", *PLOT_CONFIGS.keys()],
        default="all",
    )
    args = parser.parse_args()

    baseline_csv = Path(args.baseline_csv)
    output_dir = Path(args.output_dir)
    cases = PLOT_CONFIGS.keys() if args.case == "all" else [args.case]
    for case_name in cases:
        pdf_path, png_path = plot_case(case_name, baseline_csv, output_dir)
        print(f"Saved {case_name}: {pdf_path}")
        print(f"Saved {case_name}: {png_path}")


if __name__ == "__main__":
    main()
