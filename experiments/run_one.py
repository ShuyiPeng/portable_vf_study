import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.designs import make_design
from vf_core.runner import run_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["NL", "SP"], default="NL")
    parser.add_argument("--num-pv", type=int, default=0)
    parser.add_argument("--num-batt", type=int, default=0)
    parser.add_argument("--num-wind", type=int, default=0)
    parser.add_argument("--num-light", type=int, default=14812)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    design = make_design(args.num_pv, args.num_batt, args.num_wind, args.num_light)
    results, _ = run_simulation(args.scenario, design, quiet=not args.verbose)
    print(results["result_summary"])


if __name__ == "__main__":
    main()
