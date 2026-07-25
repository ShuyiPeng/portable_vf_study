import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.designs import make_design
from vf_core.paths import OUTPUT_DIR
from vf_core.runner import build_eval_params, load_scenario_arrays, load_weather_inputs
from legacy.simulators.sp_simulator import IntegratedSimulation as SPIntegratedSimulation


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
})


MEASUREMENT_LABELS = [
    r"Dry mass",
    r"CO$_2$ concentration",
    r"Temperature",
    r"Relative humidity",
]
MEASUREMENT_UNITS = [
    r"kg m$^{-2}$",
    r"ppm",
    r"$^\circ$C",
    r"%",
]


def style_axis(ax):
    ax.grid(False)
    ax.tick_params(direction="out", which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def roll_from_start(values, start_index: int):
    return np.roll(values, -start_index, axis=0)


def run_single_cycle(
    scenario_name: str,
    design: dict,
    start_day: float,
    target_dry_mass: float,
):
    scenario, pv_power_traj, wind_power_traj, electricity_prices, period_codes = load_scenario_arrays(
        scenario_name
    )
    if scenario.name != "SP":
        raise ValueError("This script currently supports the SP simulator.")

    h_seconds = 10 * 60
    steps_per_day = int(24 * 3600 / h_seconds)
    start_index = int(round(start_day * steps_per_day))

    pv_power_traj = roll_from_start(pv_power_traj, start_index)
    wind_power_traj = roll_from_start(wind_power_traj, start_index)
    electricity_prices = roll_from_start(electricity_prices, start_index)
    shifted_d_values = roll_from_start(load_weather_inputs(scenario.name), start_index)
    shifted_period_codes = None
    if period_codes is not None:
        shifted_period_codes = roll_from_start(period_codes, start_index)

    sim = SPIntegratedSimulation(
        16.99,
        design,
        build_eval_params(scenario.name),
        pv_power_traj,
        wind_power_traj,
        electricity_prices,
        shifted_period_codes,
    )
    sim.epw_path = str(scenario.weather_path)
    sim.output_dir = OUTPUT_DIR
    sim.save_outputs = False
    sim.external_d_values = shifted_d_values
    sim.verbose = False
    sim.run()

    dry_mass = sim.x_values[:, 0]
    reached = np.flatnonzero(dry_mass >= target_dry_mass)
    if len(reached) == 0:
        stop_index = len(dry_mass) - 1
    else:
        stop_index = max(int(reached[0]) - 1, 0)

    idx = np.arange(stop_index + 1)
    time_days = idx / steps_per_day
    return sim, time_days, stop_index


def plot_state_trajectories(time_days, y_values, output_dir: Path, stem: str):
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.3), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.25, top=0.82, wspace=0.58)
    for ax, values, label, unit in zip(axes, y_values.T, MEASUREMENT_LABELS, MEASUREMENT_UNITS):
        ax.plot(time_days, values, color="#1F78B4")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel(f"{label}\n({unit})", fontsize=9, rotation=90, labelpad=8)
        ax.tick_params(labelsize=8)
        style_axis(ax)
    return save_figure(fig, output_dir, stem)


def plot_state_soc_trajectories(time_days, y_values, soc_values, output_dir: Path, stem: str):
    fig = plt.figure(figsize=(11.0, 4.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    soc_ax = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13, top=0.91, wspace=0.58, hspace=0.55)

    for ax, values, label, unit in zip(axes, y_values.T, MEASUREMENT_LABELS, MEASUREMENT_UNITS):
        ax.plot(time_days, values, color="#1F78B4")
        ax.set_xlabel("Time (days)", fontsize=9)
        ax.set_ylabel(f"{label}\n({unit})", fontsize=9, rotation=90, labelpad=8)
        ax.tick_params(labelsize=8)
        style_axis(ax)
    soc_ax.plot(time_days, soc_values, color="#6A3D9A")
    soc_ax.set_ylabel("SOC\n(-)", fontsize=9, rotation=90, labelpad=8)
    soc_ax.set_xlabel("Time (days)", fontsize=9)
    soc_ax.set_ylim(-0.02, 1.02)
    soc_ax.tick_params(labelsize=8)
    style_axis(soc_ax)
    return save_figure(fig, output_dir, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="SP")
    parser.add_argument("--num-pv", type=int, default=3988)
    parser.add_argument("--num-batt", type=int, default=19)
    parser.add_argument("--num-wind", type=int, default=4)
    parser.add_argument("--num-light", type=int, default=14812)
    parser.add_argument("--start-day", type=float, default=171.0)
    parser.add_argument("--target-dry-mass", type=float, default=0.229)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "operation_plots"))
    parser.add_argument("--prefix", default="SP_fixed_light_14812_topsis")
    args = parser.parse_args()

    design = make_design(args.num_pv, args.num_batt, args.num_wind, args.num_light)
    sim, time_days, stop_index = run_single_cycle(
        args.scenario,
        design,
        args.start_day,
        args.target_dry_mass,
    )

    x_values = sim.x_values[: stop_index + 1, :]
    y_values = sim.y_values[: stop_index + 1, :]
    soc_values = sim.SOC_history[: stop_index + 1, 0]
    output_dir = Path(args.output_dir)

    csv_path = output_dir / f"{args.prefix}_day171_single_cycle_state_soc.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time_days": time_days,
        "x_dry_mass_kg_m2": x_values[:, 0],
        "x_co2_kg_m3": x_values[:, 1],
        "x_temperature_C": x_values[:, 2],
        "x_absolute_humidity_kg_m3": x_values[:, 3],
        "y_dry_mass_kg_m2": y_values[:, 0],
        "y_co2_ppm": y_values[:, 1],
        "y_temperature_C": y_values[:, 2],
        "y_relative_humidity_percent": y_values[:, 3],
        "soc": soc_values,
    }).to_csv(csv_path, index=False)

    generated = [
        plot_state_trajectories(
            time_days,
            y_values,
            output_dir,
            f"{args.prefix}_day171_single_cycle_x_trajectories",
        ),
        plot_state_soc_trajectories(
            time_days,
            y_values,
            soc_values,
            output_dir,
            f"{args.prefix}_day171_single_cycle_x_soc_trajectories",
        ),
    ]

    print(f"Design: {design}")
    print(f"Saved data: {csv_path}")
    print(f"Cycle duration: {time_days[-1]:.3f} days")
    print(f"Last dry mass before target: {x_values[-1, 0]:.6f} kg m^-2")
    for pdf_path, png_path in generated:
        print(pdf_path)
        print(png_path)


if __name__ == "__main__":
    main()
