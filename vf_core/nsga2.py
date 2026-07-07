from __future__ import annotations

from pathlib import Path
from typing import Optional
from multiprocessing.pool import ThreadPool
from threading import Lock

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import IntegerSBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination

try:
    from pymoo.core.problem import StarmapParallelization
except Exception:  # pragma: no cover - depends on pymoo version
    StarmapParallelization = None

from .designs import LIGHT_VALUES, make_design, make_design_from_optimizer_vector
from .objectives import default_constraints, normalized_objectives
from .paths import OUTPUT_DIR
from .runner import run_simulation

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None


DEFAULT_BOUNDS = {
    "NL": np.array([[0, 2278], [0, 500], [0, 5], [0, len(LIGHT_VALUES) - 1]], dtype=int),
    "SP": np.array([[0, 3988], [0, 500], [0, 5], [0, len(LIGHT_VALUES) - 1]], dtype=int),
}


class VFDesignProblem(ElementwiseProblem):
    def __init__(
        self,
        scenario_name: str,
        dli: float = 16.99,
        bounds: Optional[np.ndarray] = None,
        eval_overrides: Optional[dict] = None,
        min_cash_flow: float = -50000.0,
        electricity_price_scale: float = 1.0,
        grid_export_limit_w: Optional[float] = None,
        elementwise_runner=None,
    ):
        scenario_key = scenario_name.upper()
        bounds = DEFAULT_BOUNDS[scenario_key] if bounds is None else np.asarray(bounds, dtype=int)
        kwargs = {}
        if elementwise_runner is not None:
            kwargs["elementwise_runner"] = elementwise_runner
        super().__init__(
            n_var=4,
            n_obj=2,
            n_constr=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
            vtype=int,
            **kwargs,
        )
        self.scenario_name = scenario_key
        self.dli = dli
        self.eval_overrides = eval_overrides or {}
        self.min_cash_flow = min_cash_flow
        self.electricity_price_scale = electricity_price_scale
        self.grid_export_limit_w = grid_export_limit_w
        self.eval_cache = {}
        self._cache_lock = Lock()
        self.progress = None

    def _evaluate(self, x, out, *args, **kwargs):
        x_rounded = np.round(x).astype(int)
        x_rounded[3] = int(np.clip(x_rounded[3], 0, len(LIGHT_VALUES) - 1))
        key = tuple(int(v) for v in x_rounded)

        with self._cache_lock:
            cache_item = self.eval_cache.get(key)

        if cache_item is None:
            design = make_design_from_optimizer_vector(x_rounded)
            results, _ = run_simulation(
                self.scenario_name,
                design,
                eval_overrides=self.eval_overrides,
                dli=self.dli,
                quiet=True,
                electricity_price_scale=self.electricity_price_scale,
                grid_export_limit_w=self.grid_export_limit_w,
            )
            cache_item = {
                "design": design,
                "results": results,
            }
            with self._cache_lock:
                self.eval_cache[key] = cache_item

        results = cache_item["results"]
        out["F"] = normalized_objectives(results)
        out["G"] = default_constraints(results, min_cash_flow=self.min_cash_flow)
        if self.progress is not None:
            self.progress.update(1)


class VFFixedLightDesignProblem(ElementwiseProblem):
    def __init__(
        self,
        scenario_name: str,
        num_light: int = 14812,
        dli: float = 16.99,
        bounds: Optional[np.ndarray] = None,
        eval_overrides: Optional[dict] = None,
        min_cash_flow: float = -50000.0,
        electricity_price_scale: float = 1.0,
        grid_export_limit_w: Optional[float] = None,
        elementwise_runner=None,
    ):
        scenario_key = scenario_name.upper()
        full_bounds = DEFAULT_BOUNDS[scenario_key] if bounds is None else np.asarray(bounds, dtype=int)
        bounds = full_bounds[:3, :]
        kwargs = {}
        if elementwise_runner is not None:
            kwargs["elementwise_runner"] = elementwise_runner
        super().__init__(
            n_var=3,
            n_obj=2,
            n_constr=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
            vtype=int,
            **kwargs,
        )
        self.scenario_name = scenario_key
        self.num_light = int(num_light)
        self.dli = dli
        self.eval_overrides = eval_overrides or {}
        self.min_cash_flow = min_cash_flow
        self.electricity_price_scale = electricity_price_scale
        self.grid_export_limit_w = grid_export_limit_w
        self.eval_cache = {}
        self._cache_lock = Lock()
        self.progress = None

    def _evaluate(self, x, out, *args, **kwargs):
        x_rounded = np.round(x).astype(int)
        key = tuple(int(v) for v in x_rounded)

        with self._cache_lock:
            cache_item = self.eval_cache.get(key)

        if cache_item is None:
            design = make_design(
                num_pv=x_rounded[0],
                num_batt=x_rounded[1],
                num_wind=x_rounded[2],
                num_light=self.num_light,
            )
            results, _ = run_simulation(
                self.scenario_name,
                design,
                eval_overrides=self.eval_overrides,
                dli=self.dli,
                quiet=True,
                electricity_price_scale=self.electricity_price_scale,
                grid_export_limit_w=self.grid_export_limit_w,
            )
            cache_item = {
                "design": design,
                "results": results,
            }
            with self._cache_lock:
                self.eval_cache[key] = cache_item

        results = cache_item["results"]
        out["F"] = normalized_objectives(results)
        out["G"] = default_constraints(results, min_cash_flow=self.min_cash_flow)
        if self.progress is not None:
            self.progress.update(1)


def run_nsga2(
    scenario_name: str,
    pop_size: int = 120,
    n_gen: int = 20,
    seed: int = 1,
    output_name: Optional[str] = None,
    eval_overrides: Optional[dict] = None,
    min_cash_flow: float = -50000.0,
    electricity_price_scale: float = 1.0,
    grid_export_limit_w: Optional[float] = None,
    n_procs: int = 1,
):
    scenario_key = scenario_name.upper()
    pool = None
    runner = None
    if n_procs and n_procs > 1:
        if StarmapParallelization is None:
            print("Parallel runner is not available in this pymoo version; falling back to serial.")
        else:
            pool = ThreadPool(n_procs)
            runner = StarmapParallelization(pool.starmap)

    problem = VFDesignProblem(
        scenario_key,
        eval_overrides=eval_overrides,
        min_cash_flow=min_cash_flow,
        electricity_price_scale=electricity_price_scale,
        grid_export_limit_w=grid_export_limit_w,
        elementwise_runner=runner,
    )
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=IntegerSBX(prob=0.9),
        mutation=PolynomialMutation(eta=15),
        eliminate_duplicates=True,
    )
    progress_total = pop_size * n_gen
    if tqdm is not None:
        progress = tqdm(total=progress_total, desc=f"{scenario_key} NSGA-II evals", unit="sim")
    else:
        progress = None
        print(f"Running {scenario_key} NSGA-II: about {progress_total} simulations")

    problem.progress = progress
    try:
        res = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", n_gen),
            seed=seed,
            save_history=False,
            verbose=False,
        )
    finally:
        problem.progress = None
        if progress is not None:
            progress.close()
        if pool is not None:
            pool.close()
            pool.join()

    if res.F is None or res.X is None or len(res.X) == 0:
        raise ValueError("No feasible solutions found. Try relaxing constraints or increasing generations.")

    rows = []
    for x, f in zip(np.round(res.X).astype(int), res.F):
        x = x.copy()
        x[3] = int(np.clip(x[3], 0, len(LIGHT_VALUES) - 1))
        cache_item = problem.eval_cache[tuple(int(v) for v in x)]
        row = dict(cache_item["design"])
        row["num_light_index"] = int(x[3])
        row["grid_co2_emission"] = float(f[0])
        row["annual_net_cash_flow"] = float(-f[1])
        row.update(cache_item["results"]["result_summary"])
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name:
        output_path = Path(OUTPUT_DIR) / f"{output_name}.csv"
        df.to_csv(output_path, index=False)
    return df, res


def run_fixed_light_nsga2(
    scenario_name: str,
    num_light: int = 14812,
    pop_size: int = 120,
    n_gen: int = 20,
    seed: int = 1,
    output_name: Optional[str] = None,
    eval_overrides: Optional[dict] = None,
    min_cash_flow: float = -50000.0,
    electricity_price_scale: float = 1.0,
    grid_export_limit_w: Optional[float] = None,
    n_procs: int = 1,
):
    scenario_key = scenario_name.upper()
    pool = None
    runner = None
    if n_procs and n_procs > 1:
        if StarmapParallelization is None:
            print("Parallel runner is not available in this pymoo version; falling back to serial.")
        else:
            pool = ThreadPool(n_procs)
            runner = StarmapParallelization(pool.starmap)

    problem = VFFixedLightDesignProblem(
        scenario_key,
        num_light=num_light,
        eval_overrides=eval_overrides,
        min_cash_flow=min_cash_flow,
        electricity_price_scale=electricity_price_scale,
        grid_export_limit_w=grid_export_limit_w,
        elementwise_runner=runner,
    )
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=IntegerSBX(prob=0.9),
        mutation=PolynomialMutation(eta=15),
        eliminate_duplicates=True,
    )
    progress_total = pop_size * n_gen
    if tqdm is not None:
        progress = tqdm(total=progress_total, desc=f"{scenario_key} fixed-light NSGA-II evals", unit="sim")
    else:
        progress = None
        print(f"Running {scenario_key} fixed-light NSGA-II: about {progress_total} simulations")

    problem.progress = progress
    try:
        res = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", n_gen),
            seed=seed,
            save_history=False,
            verbose=False,
        )
    finally:
        problem.progress = None
        if progress is not None:
            progress.close()
        if pool is not None:
            pool.close()
            pool.join()

    if res.F is None or res.X is None or len(res.X) == 0:
        raise ValueError("No feasible solutions found. Try relaxing constraints or increasing generations.")

    rows = []
    for x, f in zip(np.round(res.X).astype(int), res.F):
        cache_item = problem.eval_cache[tuple(int(v) for v in x)]
        row = dict(cache_item["design"])
        row["grid_co2_emission"] = float(f[0])
        row["annual_net_cash_flow"] = float(-f[1])
        row.update(cache_item["results"]["result_summary"])
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name:
        output_path = Path(OUTPUT_DIR) / f"{output_name}.csv"
        df.to_csv(output_path, index=False)
    return df, res
