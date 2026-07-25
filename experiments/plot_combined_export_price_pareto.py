import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.paths import OUTPUT_DIR
from experiments.plot_experiment_pareto import (
    load_topsis_point,
    objective_values,
    resolve_csv_path,
)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "font.size": 14,
    "axes.labelsize": 15.4,
    "legend.fontsize": 11.2,
    "xtick.labelsize": 12.6,
    "ytick.labelsize": 12.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
})


CASES = [
    {
        "panel": "(a)",
        "series": [
            {"label": "SP-S1", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Export 10%", "file": "SP_export_010.csv", "marker": "^", "color": "#33A02C"},
            {"label": "Export 50%", "file": "SP_export_050.csv", "marker": "s", "color": "#E31A1C"},
        ],
    },
    {
        "panel": "(b)",
        "series": [
            {"label": "SP-S1", "file": None, "marker": "o", "color": "#1F78B4"},
            {"label": "Price 80%", "file": "SP_price_080.csv", "marker": "^", "color": "#33A02C"},
            {"label": "Price 120%", "file": "SP_price_120.csv", "marker": "s", "color": "#E31A1C"},
        ],
    },
]


def legend_elements(series_list: list[dict]) -> list[Line2D]:
    elements = []
    for series in series_list:
        elements.append(
            Line2D(
                [0],
                [0],
                marker=series["marker"],
                color="w",
                label=series["label"],
                markeredgecolor=series["color"],
                markerfacecolor="none",
                markersize=8.4,
            )
        )
    elements.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="TOPSIS-optimal",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=12.6,
        )
    )
    return elements


def plot_case(ax, case: dict, baseline_csv: Path, output_dir: Path):
    for series in case["series"]:
        csv_file = baseline_csv if series["file"] is None else resolve_csv_path(series["file"], output_dir)
        df = pd.read_csv(csv_file)
        co2, cash_million = objective_values(df)

        ax.scatter(
            co2,
            cash_million,
            marker=series["marker"],
            s=25.2,
            edgecolors=series["color"],
            facecolors="none",
            linewidths=0.8,
            alpha=0.45,
            label="_nolegend_",
        )

        topsis_co2, topsis_cash_million, _ = load_topsis_point(csv_file, co2, cash_million)
        ax.scatter(
            topsis_co2,
            topsis_cash_million,
            marker="*",
            s=161,
            color=series["color"],
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label="_nolegend_",
        )

    ax.set_xlabel(r"Grid CO$_2$ Emissions (ton/year)")
    ax.grid(False)
    ax.tick_params(direction="out", which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.legend(
        handles=legend_elements(case["series"]),
        loc="lower right",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        borderpad=0.45,
        handletextpad=0.5,
    )


def main():
    output_dir = OUTPUT_DIR
    baseline_csv = output_dir / "SP_baseline.csv"

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.96, wspace=0.24)

    for ax, case in zip(axes, CASES):
        plot_case(ax, case, baseline_csv, output_dir)
    axes[0].set_ylabel("Annual Net Profit (10⁶ EUR/year)")
    axes[1].set_ylabel("Annual Net Profit (10⁶ EUR/year)")

    for ax, case in zip(axes, CASES):
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.04, case["panel"], ha="center", va="bottom", fontsize=15.4)

    pdf_path = output_dir / "SP_export_price_pareto_combined.pdf"
    png_path = output_dir / "SP_export_price_pareto_combined.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
