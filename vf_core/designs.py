import numpy as np
from typing import Dict


LIGHT_VALUES = np.arange(14812, 21483, 741, dtype=int)


def light_index_to_num_light(index: int) -> int:
    index = int(np.clip(index, 0, len(LIGHT_VALUES) - 1))
    return int(LIGHT_VALUES[index])


def make_design(num_pv: int, num_batt: int, num_wind: int, num_light: int) -> Dict[str, int]:
    return {
        "num_pv": int(num_pv),
        "num_batt": int(num_batt),
        "num_wind": int(num_wind),
        "num_light": int(num_light),
    }


def make_design_from_optimizer_vector(x) -> Dict[str, int]:
    values = np.round(x).astype(int)
    return make_design(
        num_pv=values[0],
        num_batt=values[1],
        num_wind=values[2],
        num_light=light_index_to_num_light(values[3]),
    )
