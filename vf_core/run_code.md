# PowerShell run commands for the experiments

## 1. Activate the conda environment
```powershell
conda activate portable_vf_study
```

## 2. Go to the project root
```powershell
cd D:\portable_vf_study-main\portable_vf_study-main
```

## 3. Run the main NSGA-II experiments
```powershell
python .\experiments\run_nsga2.py --scenario NL --pop-size 120 --n-gen 20 --output-name NL_reprod_0 --n-procs 4
```

```powershell
python .\experiments\run_nsga2.py --scenario SP --pop-size 120 --n-gen 20 --output-name SP_reprod_0 --n-procs 4
```

## 4. Run the fixed-light experiment
```powershell
python .\experiments\run_fixed_light_nsga2.py --scenario SP --pop-size 120 --n-gen 20 --output-name SP_fixed_light_14812 --n-procs 4
```

```powershell
python .\experiments\run_fixed_light_nsga2.py --scenario SP --num-light 18517 --pop-size 120 --n-gen 20 --output-name SP_fixed_light_18517 --n-procs 4
```

## 5. Run the export-limit experiment
```powershell
python .\experiments\run_export_limit_nsga2.py --scenario SP --export-case 010 --pop-size 120 --n-gen 20 --output-name SP_export_010 --n-procs 4
```

```powershell
python .\experiments\run_export_limit_nsga2.py --scenario SP --export-case 050 --pop-size 120 --n-gen 20 --output-name SP_export_050 --n-procs 4
```

## 6. Run the price-sensitivity experiment
```powershell
python .\experiments\run_price_sensitivity_nsga2.py --scenario SP --price-scale 0.8 --pop-size 120 --n-gen 20 --output-name SP_price_080 --n-procs 4
```

```powershell
python .\experiments\run_price_sensitivity_nsga2.py --scenario SP --price-scale 1.2 --pop-size 120 --n-gen 20 --output-name SP_price_120 --n-procs 4
```

## 7. Run a single scenario manually
```powershell
python .\experiments\run_one.py --scenario SP --output-name SP_single_run
```

## 8. Plot Pareto results
```powershell
python .\experiments\plot_experiment_pareto.py --baseline-csv outputs\SP_baseline.csv --case export
```

```powershell
python .\experiments\plot_experiment_pareto.py --baseline-csv outputs\SP_baseline.csv --case fixed_light
```

Plot selected fixed-light Pareto CSV files in one comparison figure:
```powershell
python .\experiments\plot_experiment_pareto.py --baseline-csv outputs\SP_baseline.csv --case fixed_light --fixed-light-files outputs\SP_fixed_light_14812.csv outputs\SP_fixed_light_18517.csv --fixed-light-labels "Fixed-light 14812" "Fixed-light 18517" --output-stem SP_fixed_light_pareto_compare
```

```powershell
python .\experiments\plot_experiment_pareto.py --baseline-csv outputs\SP_baseline.csv --case price
```

## 9. Extract the TOPSIS point from a CSV
```powershell
python .\experiments\extract_topsis_point.py --input-csv outputs\SP_fixed_light_14812.csv
```

```powershell
python .\experiments\extract_topsis_point.py --input-csv outputs\SP_export_limit.csv --output-csv outputs\SP_export_limit_topsis.csv
```

## 10. Plot design variables against emissions
Plot one Pareto CSV:
```powershell
python .\experiments\plot_design_variables.py --csv outputs\SP_fixed_light_14812.csv
```

Plot the baseline Pareto CSV:
```powershell
python .\experiments\plot_design_variables.py --csv outputs\SP_baseline.csv --output-stem SP_baseline_design_variables
```

Plot baseline and fixed-light results in one figure:
```powershell
python .\experiments\plot_design_variables.py --csv outputs\SP_baseline.csv outputs\SP_fixed_light_14812.csv --labels "Co-design solutions" "Fixed-light solutions" --output-stem SP_baseline_fixed_light_design_variables_compare
```

## 11. Plot operation profiles from one design
Run the default SP-S1 design used in the paper figures:
```powershell
python .\experiments\plot_simulation_operation.py
```

Run a custom design:
```powershell
python .\experiments\plot_simulation_operation.py --scenario SP --num-pv 3987 --num-batt 218 --num-wind 5 --num-light 14812 --prefix SP_S1
```

Change the three-day windows if needed:
```powershell
python .\experiments\plot_simulation_operation.py --summer-start-day 172 --winter-start-day 15
```

## 12. If conda is not found in PowerShell
```powershell
& "D:\Anacoda\condabin\conda.bat" activate portable_vf_study
```
