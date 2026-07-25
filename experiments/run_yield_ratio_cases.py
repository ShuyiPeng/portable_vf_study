from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.plot_simulation_operation import simulation_dataframe
from vf_core.designs import make_design
from vf_core.paths import OUTPUT_DIR
from vf_core.runner import run_simulation


CASES = (
    ("NL", "renewable", 2277, 0, 5),
    ("SP", "renewable", 3985, 0, 3),
    ("NL", "grid_only", 0, 0, 0),
    ("SP", "grid_only", 0, 0, 0),
)
NUM_LIGHT = 14812


def main():
    output_dir = OUTPUT_DIR / "requested_simulations"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario, case, num_pv, num_batt, num_wind in CASES:
        design = make_design(num_pv, num_batt, num_wind, NUM_LIGHT)
        results, sim = run_simulation(scenario, design, quiet=True)

        timeseries = simulation_dataframe(sim)
        timeseries.insert(0, "scenario", scenario)
        timeseries.insert(1, "case", case)
        timeseries_path = output_dir / f"{scenario}_{case}_timeseries.csv"
        timeseries.to_csv(timeseries_path, index=False)

        fresh_yield_kg = float(
            np.sum(sim.harvest_dw_arr) * sim.p["dw_fw"] * sim.p["A_cul"]
        )
        emission_ton = float(results["co2_emission_grid"])
        electricity_cost_eur = float(sim.econ.grid_operational_cost())
        rows.append(
            {
                "scenario": scenario,
                "case": case,
                "num_pv": num_pv,
                "num_wind": num_wind,
                "num_batt": num_batt,
                "num_light": NUM_LIGHT,
                "annual_fresh_yield_kg": fresh_yield_kg,
                "annual_grid_emission_ton_co2": emission_ton,
                "annual_electricity_cost_eur": electricity_cost_eur,
                "annual_electricity_consumption_kwh_per_kg_fresh_weight": (
                    float(np.sum(sim.E_load_history)) / fresh_yield_kg
                    if fresh_yield_kg
                    else np.nan
                ),
                "emission_intensity_kg_co2eq_per_kg_fresh_yield": (
                    emission_ton * 1000.0 / fresh_yield_kg
                    if fresh_yield_kg
                    else np.nan
                ),
                "electricity_cost_per_fresh_yield_eur_per_kg": (
                    electricity_cost_eur / fresh_yield_kg
                    if fresh_yield_kg
                    else np.nan
                ),
                "annual_grid_import_kwh": float(np.sum(sim.E_grid_in_history)),
                "annual_grid_export_kwh": float(np.sum(np.abs(sim.E_grid_ex_history))),
                "timeseries_csv": str(timeseries_path),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "annual_yield_ratios_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
