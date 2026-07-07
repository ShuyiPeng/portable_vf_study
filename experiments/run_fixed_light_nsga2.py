import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.nsga2 import run_fixed_light_nsga2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["NL", "SP"], default="SP")
    parser.add_argument("--num-light", type=int, default=14812)
    parser.add_argument("--pop-size", type=int, default=120)
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--n-procs", type=int, default=1)
    args = parser.parse_args()

    output_name = args.output_name or f"{args.scenario}_fixed_light_{args.num_light}"
    df, _ = run_fixed_light_nsga2(
        args.scenario,
        num_light=args.num_light,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        output_name=output_name,
        n_procs=args.n_procs,
    )
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
