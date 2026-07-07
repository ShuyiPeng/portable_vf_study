from typing import Dict, List


CO2_SCALE = 1e4
CASH_FLOW_SCALE = 1e6


def normalized_objectives(results: Dict) -> List[float]:
    return [
        results["co2_emission_grid"] / CO2_SCALE,
        -results["obj_annual_net_cash_flow"] / CASH_FLOW_SCALE,
    ]


def default_constraints(results: Dict, min_cash_flow: float = 0.0) -> List[float]:
    cash_flow_norm = results["obj_annual_net_cash_flow"] / CASH_FLOW_SCALE
    return [
        min_cash_flow / CASH_FLOW_SCALE - cash_flow_norm,
    ]


def denormalize_pareto_frame(df):
    df = df.copy()
    df["grid_co2_emission_raw"] = df["grid_co2_emission"] * CO2_SCALE
    df["annual_net_cash_flow_raw"] = df["annual_net_cash_flow"] * CASH_FLOW_SCALE
    return df
