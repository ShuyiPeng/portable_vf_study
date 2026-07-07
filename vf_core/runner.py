from __future__ import annotations

from collections.abc import Mapping
from typing import Optional
import warnings

import numpy as np
import pvlib

from legacy.simulators.nl_simulator import IntegratedSimulation as NLIntegratedSimulation
from legacy.simulators.sp_simulator import IntegratedSimulation as SPIntegratedSimulation

from .paths import OUTPUT_DIR
from .scenarios import get_scenario


DEFAULT_EVAL_PARAMS = {
    "pv_unit_cost": 810 * 0.95 * 0.34,
    "wind_unit_cost": 1590 * 0.95 * 300,
    "batt_unit_cost": 353 * 4.8,
    "led_unit_cost": 30,
    "lettuce_price": 1.08,
}

_SCENARIO_ARRAY_CACHE = {}
_WEATHER_INPUT_CACHE = {}


def load_scenario_arrays(scenario_name: str):
    scenario_key = scenario_name.upper()
    if scenario_key in _SCENARIO_ARRAY_CACHE:
        return _SCENARIO_ARRAY_CACHE[scenario_key]

    scenario = get_scenario(scenario_name)
    pv_power_traj = np.load(scenario.pv_power_path)
    wind_power_traj = np.load(scenario.wind_power_path)
    electricity_prices = np.load(scenario.electricity_price_path)
    period_codes = None
    if scenario.period_codes_path is not None:
        period_codes = np.load(scenario.period_codes_path).astype(np.int8)
    loaded = (scenario, pv_power_traj, wind_power_traj, electricity_prices, period_codes)
    _SCENARIO_ARRAY_CACHE[scenario_key] = loaded
    return loaded


def load_weather_inputs(scenario_name: str):
    scenario = get_scenario(scenario_name)
    if scenario.name in _WEATHER_INPUT_CACHE:
        return _WEATHER_INPUT_CACHE[scenario.name]

    data, _ = pvlib.iotools.read_epw(scenario.weather_path)
    selected_data = data[["temp_air", "ghi", "wind_speed"]].copy()
    selected_data_10min = (
        selected_data
        .resample("10min")
        .interpolate(method="linear", limit_direction="both")
    )
    tem_data = selected_data_10min["temp_air"].values
    solar_data = selected_data_10min["ghi"].values
    d_values = np.column_stack((solar_data, tem_data))
    _WEATHER_INPUT_CACHE[scenario.name] = d_values
    return d_values


def build_eval_params(scenario_name: str, overrides: Optional[Mapping] = None) -> dict:
    scenario = get_scenario(scenario_name)
    params = dict(DEFAULT_EVAL_PARAMS)
    params["co2_emission_factor"] = scenario.co2_emission_factor
    if overrides:
        params.update(overrides)
    return params


def run_simulation(
    scenario_name: str,
    design_para: Mapping,
    eval_overrides: Optional[Mapping] = None,
    dli: float = 16.99,
    output_dir=None,
    quiet: bool = True,
    save_outputs: bool = False,
    electricity_price_scale: float = 1.0,
    grid_export_limit_w: Optional[float] = None,
):
    scenario, pv_power_traj, wind_power_traj, electricity_prices, period_codes = load_scenario_arrays(
        scenario_name
    )
    if electricity_price_scale != 1.0:
        electricity_prices = electricity_prices.copy() * electricity_price_scale
    new_eval_p = build_eval_params(scenario.name, eval_overrides)
    output_dir = OUTPUT_DIR if output_dir is None else output_dir

    if scenario.name == "NL":
        sim = NLIntegratedSimulation(
            dli,
            dict(design_para),
            new_eval_p,
            pv_power_traj,
            wind_power_traj,
            electricity_prices,
            period_codes,
        )
    elif scenario.name == "SP":
        sim = SPIntegratedSimulation(
            dli,
            dict(design_para),
            new_eval_p,
            pv_power_traj,
            wind_power_traj,
            electricity_prices,
            period_codes,
        )
    else:
        raise ValueError(f"Unsupported scenario {scenario.name!r}.")

    sim.epw_path = str(scenario.weather_path)
    sim.output_dir = output_dir
    sim.save_outputs = save_outputs
    sim.external_d_values = load_weather_inputs(scenario.name)
    sim.verbose = not quiet
    if grid_export_limit_w is not None:
        sim.batt_param["P_grid_max_export"] = float(grid_export_limit_w)

    if quiet:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            results = sim.run()
    else:
        results = sim.run()
    return results, sim
