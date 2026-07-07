# Portable Vertical Farm Design Study

This folder is a minimal portable version of the simulation and NSGA-II workflow.

## Run one simulation

```bash
python experiments/run_one.py --scenario NL --num-pv 0 --num-batt 0 --num-wind 0 --num-light 14812
python experiments/run_one.py --scenario SP --num-pv 0 --num-batt 0 --num-wind 0 --num-light 14812
```

## Run a small NSGA-II smoke test

```bash
python experiments/run_nsga2.py --scenario NL --pop-size 4 --n-gen 1 --output-name nl_smoke
python experiments/run_nsga2.py --scenario SP --pop-size 4 --n-gen 1 --output-name sp_smoke
```

Large experiments should increase `--pop-size` and `--n-gen`.

