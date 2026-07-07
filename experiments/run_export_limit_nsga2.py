import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.nsga2 import run_nsga2


EXPORT_LIMITS_W = {
    "010": 0.175e6,
    "020": 0.35e6,
    "050": 0.875e6,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["NL", "SP"], default="SP")
    parser.add_argument("--export-case", choices=sorted(EXPORT_LIMITS_W), default=None)
    parser.add_argument("--export-limit-w", type=float, default=None)
    parser.add_argument("--pop-size", type=int, default=120)
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--n-procs", type=int, default=1)
    args = parser.parse_args()

    if args.export_limit_w is None and args.export_case is None:
        parser.error("Specify either --export-case or --export-limit-w.")

    if args.export_limit_w is not None:
        export_limit_w = args.export_limit_w
        export_name = f"{int(round(export_limit_w))}"
    else:
        export_limit_w = EXPORT_LIMITS_W[args.export_case]
        export_name = args.export_case

    output_name = args.output_name or f"{args.scenario}_export_{export_name}"
    df, _ = run_nsga2(
        args.scenario,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        output_name=output_name,
        grid_export_limit_w=export_limit_w,
        n_procs=args.n_procs,
    )
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
