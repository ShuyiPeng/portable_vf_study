from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import DATA_DIR


@dataclass(frozen=True)
class Scenario:
    name: str
    pv_power_file: str
    wind_power_file: str
    electricity_price_file: str
    weather_file: str
    co2_emission_factor: float
    period_codes_file: Optional[str] = None

    @property
    def pv_power_path(self) -> Path:
        return DATA_DIR / self.pv_power_file

    @property
    def wind_power_path(self) -> Path:
        return DATA_DIR / self.wind_power_file

    @property
    def electricity_price_path(self) -> Path:
        return DATA_DIR / self.electricity_price_file

    @property
    def weather_path(self) -> Path:
        return DATA_DIR / self.weather_file

    @property
    def period_codes_path(self) -> Optional[Path]:
        if self.period_codes_file is None:
            return None
        return DATA_DIR / self.period_codes_file


SCENARIOS = {
    "NL": Scenario(
        name="NL",
        pv_power_file="NL_TMY_year_pv_power_traj.npy",
        wind_power_file="NL_TMY_year_wind_power_traj.npy",
        electricity_price_file="NL_syn_electricity_prices.npy",
        weather_file="NLD_GE_Hupsel.062830_TMYx.epw",
        period_codes_file="time_period_codes_2023_spain.npy",
        co2_emission_factor=0.329,
    ),
    "SP": Scenario(
        name="SP",
        pv_power_file="SP_TMY_year_pv_power_traj.npy",
        wind_power_file="SP_TMY_year_wind_power_traj.npy",
        electricity_price_file="SP_syn_electricity_prices.npy",
        weather_file="ESP_AN_Sevilla.AP.083910_TMYx.epw",
        period_codes_file="time_period_codes_2023_spain.npy",
        co2_emission_factor=0.174,
    ),
}


def get_scenario(name: str) -> Scenario:
    key = name.upper()
    if key not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Expected one of {sorted(SCENARIOS)}.")
    return SCENARIOS[key]
