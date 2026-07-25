# Portable Vertical Farm Design Study

A portable simulation and multi-objective optimization workflow for evaluating
vertical-farm energy system designs in the Netherlands and Spain.

The project combines a time-series vertical-farm simulator with NSGA-II to
explore trade-offs between grid-related CO2 emissions and annual net cash flow.
Candidate systems can include photovoltaic panels, battery storage, wind
turbines, and different lighting capacities. The repository also contains
scripts for sensitivity studies, TOPSIS compromise-point selection, and
publication-ready or interactive visualizations.

## Contents

- [Key features](#key-features)
- [Study scenarios](#study-scenarios)
- [Optimization problem](#optimization-problem)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Optimization experiments](#optimization-experiments)
- [Post-processing and visualization](#post-processing-and-visualization)
- [Output data](#output-data)
- [Using the Python API](#using-the-python-api)
- [Reproducibility and performance](#reproducibility-and-performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Key features

- Annual time-series simulation of vertical-farm electricity demand,
  renewable generation, battery operation, grid exchange, crop production,
  operating cost, and grid emissions.
- Two bundled geographical scenarios:
  - `NL`: Hupsel, the Netherlands.
  - `SP`: Seville, Spain.
- Integer multi-objective optimization using NSGA-II.
- Variable or fixed lighting-capacity optimization.
- Electricity-price and grid-export-limit sensitivity experiments.
- Optional parallel objective evaluation.
- TOPSIS-based selection of a balanced Pareto solution.
- Static Pareto, design-variable, operational, and state-trajectory plots.
- Standalone interactive Pareto charts generated as HTML.

## Study scenarios

Scenario definitions are stored in
[`vf_core/scenarios.py`](vf_core/scenarios.py). Each scenario uses bundled
weather, renewable-power, and electricity-price time series from `data/`.

| Code | Location | Weather file | Grid CO2 factor |
| --- | --- | --- | ---: |
| `NL` | Hupsel, Netherlands | `NLD_GE_Hupsel.062830_TMYx.epw` | 0.329 |
| `SP` | Seville, Spain | `ESP_AN_Sevilla.AP.083910_TMYx.epw` | 0.174 |

The scenario data include:

- Typical-meteorological-year EPW weather files.
- Precomputed PV and wind power trajectories (`.npy`).
- Synthetic electricity-price trajectories (`.npy`).
- Electricity-period codes used by the simulator.

All paths are resolved relative to the repository, so commands should work
after cloning without editing absolute file paths.

## Optimization problem

### Design variables

| Variable | Meaning | Default optimization range |
| --- | --- | --- |
| `num_pv` | Number of PV units | `0–2278` (`NL`), `0–3988` (`SP`) |
| `num_batt` | Number of battery units | `0–500` |
| `num_wind` | Number of wind units | `0–5` |
| `num_light` | Lighting capacity/count | Discrete values from `14812` to `21482` in steps of `741` |

The fixed-light workflow holds `num_light` constant and optimizes the other
three variables.

### Objectives

NSGA-II minimizes two normalized objectives:

1. Grid-related CO2 emissions.
2. Negative annual net cash flow, which is equivalent to maximizing annual net
   cash flow.

The normalization constants are `1e4` for emissions and `1e6` for cash flow.
The default feasibility constraint requires annual net cash flow to be at least
`-50,000` in the simulator's currency units.

The Pareto CSV files contain normalized optimization columns together with the
simulator's unnormalized result summary. Use the explicit `*_raw` or descriptive
summary fields when interpreting physical or financial values.

## Repository structure

```text
portable_vf_study/
├── data/                    # Weather and precomputed scenario time series
├── experiments/             # Command-line experiments and plotting scripts
├── legacy/
│   ├── economics/           # Scenario-specific economic calculations
│   ├── models/              # PV, wind, and crop/model functions
│   ├── simulators/          # NL and SP simulation implementations
│   └── utils/               # Legacy helper functions
├── vf_core/
│   ├── designs.py           # Design construction and light discretization
│   ├── nsga2.py             # Optimization problems and NSGA-II runners
│   ├── objectives.py        # Objectives, scaling, and constraints
│   ├── paths.py             # Repository-relative paths
│   ├── runner.py            # Unified scenario simulation interface
│   └── scenarios.py         # NL and SP scenario definitions
├── outputs/                 # Generated CSV, PNG, PDF, and HTML files (ignored)
├── requirements.txt
└── README.md
```

The `legacy/` name reflects the origin of the underlying domain models; these
modules are still used by the portable interface in `vf_core/`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShuyiPeng/portable_vf_study.git
cd portable_vf_study
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Main dependencies include NumPy, pandas, SciPy, Matplotlib, seaborn, pvlib,
openpyxl, pymoo, and tqdm.

## Quick start

Run commands from the repository root.

### Run one design

Netherlands:

```bash
python experiments/run_one.py --scenario NL --num-pv 0 --num-batt 0 --num-wind 0 --num-light 14812
```

Spain:

```bash
python experiments/run_one.py --scenario SP --num-pv 0 --num-batt 0 --num-wind 0 --num-light 14812
```

Add `--verbose` to show simulator output:

```bash
python experiments/run_one.py --scenario SP --num-pv 1000 --num-batt 50 --num-wind 2 --num-light 14812 --verbose
```

### Run a smoke-test optimization

```bash
python experiments/run_nsga2.py --scenario NL --pop-size 4 --n-gen 1 --output-name nl_smoke
```

This writes `outputs/nl_smoke.csv`. A smoke test verifies that the workflow
runs; its small population and generation count are not suitable for scientific
analysis.

## Optimization experiments

### Standard four-variable NSGA-II

```bash
python experiments/run_nsga2.py \
  --scenario SP \
  --pop-size 120 \
  --n-gen 20 \
  --n-procs 1 \
  --output-name SP_baseline
```

On PowerShell, either enter the command on one line or replace each trailing
backslash with a backtick.

Options:

- `--scenario {NL,SP}` selects the study location.
- `--pop-size` controls the population size.
- `--n-gen` controls the number of generations.
- `--n-procs` controls concurrent simulation evaluations.
- `--output-name` writes `outputs/<name>.csv`; without it, results are printed
  but not saved by the standard runner.

### Fixed-light optimization

```bash
python experiments/run_fixed_light_nsga2.py --scenario SP --num-light 14812 --pop-size 120 --n-gen 20 --seed 1 --n-procs 1
```

The default output name is
`outputs/<scenario>_fixed_light_<num_light>.csv`.

### Electricity-price sensitivity

`--price-scale` multiplies the scenario electricity-price trajectory. For
example, `0.8` represents 80% of the baseline values and `1.2` represents 120%.

```bash
python experiments/run_price_sensitivity_nsga2.py --scenario SP --price-scale 0.8 --pop-size 120 --n-gen 20 --seed 1
python experiments/run_price_sensitivity_nsga2.py --scenario SP --price-scale 1.2 --pop-size 120 --n-gen 20 --seed 1
```

Default output names use the percentage scale, such as `SP_price_080.csv` and
`SP_price_120.csv`.

### Grid-export-limit sensitivity

Use a predefined export case:

```bash
python experiments/run_export_limit_nsga2.py --scenario SP --export-case 020 --pop-size 120 --n-gen 20 --seed 1
```

Predefined limits are:

| Case | Export limit |
| --- | ---: |
| `010` | 175,000 W |
| `020` | 350,000 W |
| `050` | 875,000 W |

Alternatively, provide a custom limit:

```bash
python experiments/run_export_limit_nsga2.py --scenario SP --export-limit-w 500000 --pop-size 120 --n-gen 20
```

### Yield-ratio cases

```bash
python experiments/run_yield_ratio_cases.py
```

This runs the predefined NL/SP cases in the script and exports both time-series
files and a summary CSV.

## Post-processing and visualization

Most plotting commands read Pareto CSV files from `outputs/` and write figures
back to an output directory.

### Extract a TOPSIS compromise solution

```bash
python experiments/extract_topsis_point.py --input-csv outputs/SP_baseline.csv
```

The default result is `outputs/SP_baseline_topsis.csv`. Multiple input CSV files
may be passed after `--input-csv`.

### Plot Pareto fronts

```bash
python experiments/plot_experiment_pareto.py \
  --baseline-csv outputs/SP_baseline.csv \
  --case fixed_light \
  --fixed-light-files outputs/SP_fixed_light_14812.csv \
  --fixed-light-labels "Fixed light 14812" \
  --output-stem SP_pareto
```

Run the script with `--help` for the complete set of comparison options:

```bash
python experiments/plot_experiment_pareto.py --help
```

### Plot design variables along a Pareto front

```bash
python experiments/plot_design_variables.py --csv outputs/SP_baseline.csv --labels "SP baseline"
```

Multiple CSV files and matching labels can be supplied for comparison.

### Create an interactive Pareto chart

```bash
python experiments/make_interactive_pareto.py --case fixed_light
```

Available cases are defined in `SERIES_BY_CASE` inside the script. The output is
a standalone HTML file that can be opened in a browser without a web server.

### Plot simulation operation

```bash
python experiments/plot_simulation_operation.py \
  --scenario SP \
  --num-pv 3988 \
  --num-batt 19 \
  --num-wind 4 \
  --num-light 14812 \
  --combined-figures
```

This exports an operational time series and plots such as annual renewable
generation, battery state of charge, seasonal power balance, and electricity
price windows.

### Plot single-cycle state trajectories

```bash
python experiments/plot_single_cycle_state_trajectories.py \
  --scenario SP \
  --num-pv 3988 \
  --num-batt 19 \
  --num-wind 4 \
  --num-light 14812
```

This workflow currently targets the SP simulator and visualizes crop/state and
battery trajectories for a selected starting day.

Additional combined Pareto and specialized plotting workflows are available in:

- `experiments/plot_combined_export_price_pareto.py`
- `experiments/plot_simulation_operation.py`
- `experiments/plot_single_cycle_state_trajectories.py`

Use `python <script> --help` to inspect current command-line arguments.

## Output data

Generated artifacts are stored under `outputs/` by default:

- Pareto solution tables: `.csv`
- TOPSIS-selected solutions: `*_topsis.csv`
- Simulation time series: `.csv`
- Static figures: `.png` and `.pdf`
- Interactive figures: `.html`

The directory is excluded by `.gitignore`, so generated results are not
committed to the repository by default.

Common Pareto output fields include:

| Field | Description |
| --- | --- |
| `num_pv` | Selected PV-unit count |
| `num_batt` | Selected battery-unit count |
| `num_wind` | Selected wind-unit count |
| `num_light` | Selected lighting count/capacity |
| `num_light_index` | Discrete lighting index in variable-light runs |
| `grid_co2_emission` | Normalized emissions objective |
| `annual_net_cash_flow` | Normalized cash-flow objective, reported with maximizing sign |

Additional columns come from the simulator's `result_summary` and contain the
detailed physical, crop, financial, and emissions results.

## Using the Python API

Run a simulation directly:

```python
from vf_core.designs import make_design
from vf_core.runner import run_simulation

design = make_design(
    num_pv=1000,
    num_batt=50,
    num_wind=2,
    num_light=14812,
)

results, simulator = run_simulation("SP", design, quiet=True)
print(results["result_summary"])
```

Run NSGA-II programmatically:

```python
from vf_core.nsga2 import run_nsga2

pareto, optimization_result = run_nsga2(
    "SP",
    pop_size=120,
    n_gen=20,
    seed=1,
    output_name="SP_baseline",
    n_procs=1,
)

print(pareto.head())
```

Sensitivity arguments supported by the API include:

- `electricity_price_scale`
- `grid_export_limit_w`
- `min_cash_flow`
- `eval_overrides`

## Reproducibility and performance

- Use the same `seed`, population size, generation count, scenario, and
  dependency versions when comparing optimization runs.
- A run performs approximately `pop_size × n_gen` objective evaluations,
  although duplicate elimination and the internal evaluation cache can change
  the exact number of full simulations.
- Full optimizations can be computationally expensive. Start with a small smoke
  test before increasing `--pop-size` and `--n-gen`.
- Set `--n-procs` above `1` to request parallel evaluation. Availability depends
  on the installed pymoo version; unsupported versions fall back to serial
  execution.
- The runner caches scenario arrays and weather inputs within a Python process
  to reduce repeated file-loading overhead.
- Keep raw outputs and the exact command used for each study run in a separate
  archival location because `outputs/` is intentionally ignored by Git.

## Troubleshooting

### `ModuleNotFoundError`

Run commands from the repository root and verify that the virtual environment
is active:

```bash
python -m pip install -r requirements.txt
```

### No feasible solutions found

Increase `--pop-size` or `--n-gen`. For programmatic runs, consider whether the
`min_cash_flow` constraint or experimental limits are too restrictive.

### Optimization is slow

First verify the workflow with a small run, then increase `--n-procs` if your
pymoo installation supports parallel element-wise evaluation.

### PowerShell blocks virtual-environment activation

You can use the environment's Python executable without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiments\run_one.py --scenario NL
```

### Output files do not appear in `git status`

This is intentional: `outputs/` is ignored to prevent generated and potentially
large experiment artifacts from being uploaded.

## License

This project is distributed under the terms in [`LICENSE`](LICENSE).
