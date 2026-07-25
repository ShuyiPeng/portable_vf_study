import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.designs import make_design
from vf_core.paths import OUTPUT_DIR
from vf_core.runner import run_simulation


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


DEFAULT_DESIGN = {
    "num_pv": 3987,
    "num_batt": 218,
    "num_wind": 5,
    "num_light": 14812,
}


COLORS = {
    "pv": "#FDBF6F",
    "wind": "#1F78B4",
    "load": "#222222",
    "battery": "#6A3D9A",
    "grid_import": "#E31A1C",
    "grid_export": "#33A02C",
    "curtailment": "#B15928",
    "price": "#1F78B4",
    "soc": "#6A3D9A",
}


LINESTYLES = {
    "pv": "-",
    "wind": "--",
    "battery": "-",
    "grid_import": "-.",
    "grid_export": ":",
    "curtailment": "-",
    "load": "-",
}


def flat(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def simulation_dataframe(sim) -> pd.DataFrame:
    n = int(sim.N)
    time_days = np.arange(n) * float(sim.h) / 3600.0 / 24.0
    return pd.DataFrame({
        "time_days": time_days,
        "pv_power_MW": flat(sim.pv_power_traj) / 1e6,
        "wind_power_MW": flat(sim.wind_power_traj) / 1e6,
        "load_power_MW": flat(sim.p_load) / 1e6,
        "battery_power_MW": flat(sim.P_battery_history) / 1e6,
        "grid_import_MW": flat(sim.P_grid_in_history) / 1e6,
        "grid_export_MW": flat(sim.P_grid_ex_history) / 1e6,
        "curtailment_MW": flat(sim.P_excess_history) / 1e6,
        "soc": flat(sim.SOC_history),
        "electricity_price_EUR_per_kWh": flat(sim.electricity_price_history),
    })


def save_figure(fig, output_dir: Path, stem: str, tight: bool = True) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def style_axis(ax):
    ax.grid(False)
    ax.tick_params(direction="out", which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)


def add_panel_labels(fig, axes, y: float):
    for panel_label, ax in zip(("a", "b"), axes):
        bbox = ax.get_position()
        x = (bbox.x0 + bbox.x1) / 2.0
        fig.text(x, y, f"({panel_label})", ha="center", va="bottom", fontsize=10)


def safe_print(text: str):
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def time_window(df: pd.DataFrame, start_day: float, duration_days: float) -> pd.DataFrame:
    mask = (df["time_days"] >= start_day) & (df["time_days"] < start_day + duration_days)
    out = df.loc[mask].copy()
    out["time_hour"] = out["time_days"] * 24.0
    return out


def daily_generation(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["day"] = np.floor(work["time_days"]).astype(int)
    # MW * 10 min = MWh; convert to kWh/day for readable annual totals.
    dt_hours = 10.0 / 60.0
    return work.groupby("day")[["pv_power_MW", "wind_power_MW"]].sum() * dt_hours * 1000.0


def plot_annual_generation(df: pd.DataFrame, output_dir: Path, prefix: str):
    daily = daily_generation(df)
    fig, ax = plt.subplots(figsize=(6.8, 2.25))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.25, top=0.88)
    ax.plot(daily.index + 1, daily["pv_power_MW"], color=COLORS["pv"], label="PV generation")
    ax.plot(daily.index + 1, daily["wind_power_MW"], color=COLORS["wind"], label="Wind generation")
    ax.set_xlabel("Day of year", fontsize=9)
    ax.set_ylabel("Daily generation (kWh/day)", fontsize=9)
    ax.tick_params(labelsize=8)
    style_axis(ax)
    ax.legend(frameon=True, edgecolor="black", fancybox=False, fontsize=8)
    return save_figure(fig, output_dir, f"{prefix}_annual_pv_wind_generation", tight=False)


def plot_annual_generation_compare(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path,
    stem: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.24, top=0.75, wspace=0.16)
    for ax, (label, df) in zip(axes, datasets.items()):
        daily = daily_generation(df)
        ax.plot(daily.index + 1, daily["pv_power_MW"], color=COLORS["pv"], label="PV")
        ax.plot(daily.index + 1, daily["wind_power_MW"], color=COLORS["wind"], label="Wind")
        ax.set_xlabel("Day of year", fontsize=9)
        ax.tick_params(labelsize=8)
        style_axis(ax)
    axes[0].set_ylabel("Daily generation (kWh/day)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
    )
    return save_figure(fig, output_dir, stem, tight=False)


def plot_soc_window(df: pd.DataFrame, output_dir: Path, prefix: str, start_day: float, label: str):
    window = time_window(df, start_day, 3.0)
    fig, ax = plt.subplots()
    ax.plot(window["time_hour"], window["soc"], color=COLORS["soc"])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Battery state of charge")
    ax.set_ylim(-0.02, 1.02)
    style_axis(ax)
    return save_figure(fig, output_dir, f"{prefix}_{label}_three_day_battery_soc")


def plot_power_balance_window(df: pd.DataFrame, output_dir: Path, prefix: str, start_day: float, label: str):
    window = time_window(df, start_day, 3.0)
    fig, ax = plt.subplots()
    ax.plot(window["time_hour"], window["pv_power_MW"], color=COLORS["pv"], linestyle=LINESTYLES["pv"], label="PV")
    ax.plot(window["time_hour"], window["wind_power_MW"], color=COLORS["wind"], linestyle=LINESTYLES["wind"], label="Wind")
    ax.plot(window["time_hour"], window["battery_power_MW"], color=COLORS["battery"], linestyle=LINESTYLES["battery"], label="Battery")
    ax.plot(window["time_hour"], window["grid_import_MW"].abs(), color=COLORS["grid_import"], linestyle=LINESTYLES["grid_import"], label="Grid import")
    ax.plot(window["time_hour"], -window["grid_export_MW"].abs(), color=COLORS["grid_export"], linestyle=LINESTYLES["grid_export"], label="Grid export")
    ax.plot(window["time_hour"], -window["curtailment_MW"].abs(), color=COLORS["curtailment"], linestyle=LINESTYLES["curtailment"], label="Curtailment")
    ax.plot(window["time_hour"], -window["load_power_MW"].abs(), color=COLORS["load"], linestyle=LINESTYLES["load"], label="Load")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (MW)")
    style_axis(ax)
    ax.legend(
        frameon=True,
        edgecolor="black",
        fancybox=False,
        ncol=2,
        loc="best",
    )
    return save_figure(fig, output_dir, f"{prefix}_{label}_three_day_power_balance")


def plot_price_window(df: pd.DataFrame, output_dir: Path, prefix: str, start_day: float, label: str):
    window = time_window(df, start_day, 3.0)
    fig, ax = plt.subplots()
    ax.plot(
        window["time_hour"],
        window["electricity_price_EUR_per_kWh"],
        color=COLORS["price"],
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Electricity price (€/kWh)")
    style_axis(ax)
    return save_figure(fig, output_dir, f"{prefix}_{label}_three_day_electricity_price")


def plot_soc_price_window(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    start_day: float,
    label: str,
):
    window = time_window(df, start_day, 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.31, top=0.91, wspace=0.28)

    axes[0].plot(window["time_hour"], window["soc"], color=COLORS["soc"])
    axes[0].set_xlabel("Time (h)", fontsize=9)
    axes[0].set_ylabel("State of charge", fontsize=9)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].tick_params(labelsize=8)
    style_axis(axes[0])

    axes[1].plot(
        window["time_hour"],
        window["electricity_price_EUR_per_kWh"],
        color=COLORS["price"],
    )
    axes[1].set_xlabel("Time (h)", fontsize=9)
    axes[1].set_ylabel("Price (€/kWh)", fontsize=9)
    axes[1].tick_params(labelsize=8)
    style_axis(axes[1])

    add_panel_labels(fig, axes, y=0.055)
    return save_figure(fig, output_dir, f"{prefix}_{label}_three_day_soc_price", tight=False)


def plot_power_balance_season_compare(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    summer_start_day: float,
    winter_start_day: float,
):
    windows = {
        "Summer": time_window(df, summer_start_day, 3.0),
        "Winter": time_window(df, winter_start_day, 3.0),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.825), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.86, wspace=0.12)
    for ax, (label, window) in zip(axes, windows.items()):
        ax.plot(window["time_hour"], window["pv_power_MW"], color=COLORS["pv"], linestyle=LINESTYLES["pv"], label="PV")
        ax.plot(window["time_hour"], window["wind_power_MW"], color=COLORS["wind"], linestyle=LINESTYLES["wind"], label="Wind")
        ax.plot(window["time_hour"], window["battery_power_MW"], color=COLORS["battery"], linestyle=LINESTYLES["battery"], label="Battery")
        ax.plot(window["time_hour"], window["grid_import_MW"].abs(), color=COLORS["grid_import"], linestyle=LINESTYLES["grid_import"], label="Grid import")
        ax.plot(window["time_hour"], -window["grid_export_MW"].abs(), color=COLORS["grid_export"], linestyle=LINESTYLES["grid_export"], label="Grid export")
        ax.plot(window["time_hour"], -window["curtailment_MW"].abs(), color=COLORS["curtailment"], linestyle=LINESTYLES["curtailment"], label="Curtailment")
        ax.plot(window["time_hour"], -window["load_power_MW"].abs(), color=COLORS["load"], linestyle=LINESTYLES["load"], label="Load")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Time (h)", fontsize=9)
        ax.tick_params(labelsize=8)
        style_axis(ax)
    axes[0].set_ylabel("Power (MW)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        fontsize=6.8,
    )
    add_panel_labels(fig, axes, y=0.04)
    return save_figure(fig, output_dir, f"{prefix}_summer_winter_three_day_power_balance", tight=False)


def run_and_plot(
    scenario: str,
    design: dict,
    output_dir: Path,
    prefix: str,
    summer_start_day: float,
    winter_start_day: float,
):
    results, sim = run_simulation(
        scenario,
        design,
        quiet=True,
        save_outputs=False,
    )
    df = simulation_dataframe(sim)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / f"{prefix}_operation_timeseries.csv"
    df.to_csv(timeseries_path, index=False)

    generated = []
    generated.append(plot_annual_generation(df, output_dir, prefix))
    generated.append(plot_soc_window(df, output_dir, prefix, summer_start_day, "summer"))
    generated.append(plot_power_balance_window(df, output_dir, prefix, summer_start_day, "summer"))
    generated.append(plot_power_balance_window(df, output_dir, prefix, winter_start_day, "winter"))
    generated.append(plot_price_window(df, output_dir, prefix, summer_start_day, "summer"))
    return results, timeseries_path, generated


def load_or_run_timeseries(
    scenario: str,
    design: dict,
    output_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    timeseries_path = output_dir / f"{prefix}_operation_timeseries.csv"
    if timeseries_path.exists():
        return pd.read_csv(timeseries_path)
    _, sim = run_simulation(
        scenario,
        design,
        quiet=True,
        save_outputs=False,
    )
    df = simulation_dataframe(sim)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(timeseries_path, index=False)
    return df


def generate_combined_figures(
    design: dict,
    output_dir: Path,
    prefix: str,
    summer_start_day: float,
    winter_start_day: float,
):
    sp_df = load_or_run_timeseries("SP", design, output_dir, prefix)
    generated = []
    generated.append(plot_annual_generation(sp_df, output_dir, prefix))
    generated.append(plot_soc_price_window(sp_df, output_dir, prefix, summer_start_day, "summer"))
    generated.append(plot_power_balance_season_compare(
        sp_df,
        output_dir,
        prefix,
        summer_start_day,
        winter_start_day,
    ))
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["NL", "SP"], default="SP")
    parser.add_argument("--num-pv", type=int, default=DEFAULT_DESIGN["num_pv"])
    parser.add_argument("--num-batt", type=int, default=DEFAULT_DESIGN["num_batt"])
    parser.add_argument("--num-wind", type=int, default=DEFAULT_DESIGN["num_wind"])
    parser.add_argument("--num-light", type=int, default=DEFAULT_DESIGN["num_light"])
    parser.add_argument("--summer-start-day", type=float, default=171.0)
    parser.add_argument("--winter-start-day", type=float, default=15.0)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "operation_plots"))
    parser.add_argument("--prefix", default="SP_S1")
    parser.add_argument("--combined-figures", action="store_true")
    args = parser.parse_args()

    design = make_design(args.num_pv, args.num_batt, args.num_wind, args.num_light)
    results, timeseries_path, generated = run_and_plot(
        args.scenario,
        design,
        Path(args.output_dir),
        args.prefix,
        args.summer_start_day,
        args.winter_start_day,
    )

    safe_print("Simulation summary:")
    safe_print(pd.Series(results["result_summary"]).to_string())
    print(f"Saved timeseries: {timeseries_path}")
    for pdf_path, png_path in generated:
        print(f"Saved figure: {pdf_path}")
        print(f"Saved figure: {png_path}")

    if args.combined_figures:
        combined = generate_combined_figures(
            design,
            Path(args.output_dir),
            args.prefix,
            args.summer_start_day,
            args.winter_start_day,
        )
        for pdf_path, png_path in combined:
            print(f"Saved combined figure: {pdf_path}")
            print(f"Saved combined figure: {png_path}")


if __name__ == "__main__":
    main()
