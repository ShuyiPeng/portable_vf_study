import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.nsga2 import run_nsga2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["NL", "SP"], default="NL")
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--n-gen", type=int, default=1)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--n-procs", type=int, default=1)
    args = parser.parse_args()

    df, _ = run_nsga2(
        args.scenario,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        output_name=args.output_name,
        n_procs=args.n_procs,
    )
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
