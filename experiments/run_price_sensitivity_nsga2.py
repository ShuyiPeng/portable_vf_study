import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.nsga2 import run_nsga2


def format_scale_for_name(scale: float) -> str:
    return f"{int(round(scale * 100)):03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["NL", "SP"], default="SP")
    parser.add_argument("--price-scale", type=float, required=True)
    parser.add_argument("--pop-size", type=int, default=120)
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--n-procs", type=int, default=1)









    
    args = parser.parse_args()

    scale_name = format_scale_for_name(args.price_scale)
    output_name = args.output_name or f"{args.scenario}_price_{scale_name}"
    df, _ = run_nsga2(
        args.scenario,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        output_name=output_name,
        electricity_price_scale=args.price_scale,
        n_procs=args.n_procs,
    )
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
