"""Research commands for correctness checks and reproducible experiments."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .baselines import optimized_attention
from .benchmark import benchmark_local, error_metrics
from .distributed import probe_collectives
from .fixtures import AttentionConfig, make_inputs
from .partial import context_attention
from .reference import reference_attention
from .results import summarize_directory


def add_config(parser, *, lengths):
    if lengths:
        parser.add_argument(
            "--lengths",
            nargs="+",
            type=int,
            default=[4096, 16384, 65536, 131072, 262144],
        )
    else:
        parser.add_argument("--length", type=int, default=17)
    parser.add_argument("--hq", type=int, default=32)
    parser.add_argument("--hkv", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")


def config_from(args, length):
    return AttentionConfig(
        length=length,
        hq=args.hq,
        hkv=args.hkv,
        dim=args.dim,
        dtype=args.dtype,
        seed=args.seed,
        block_size=args.block_size,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="weftlm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="Check two-participant ring all-gather")
    probe.add_argument("--payload-bytes", type=int, default=16640)
    probe.add_argument("--warmup", type=int, default=3)
    probe.add_argument("--repeats", type=int, default=10)
    check = commands.add_parser("check", help="Compare synthetic attention outputs")
    add_config(check, lengths=False)
    bench = commands.add_parser("bench", help="Measure a synthetic attention operation")
    add_config(bench, lengths=True)
    bench.add_argument("--mode", choices=["single", "local-context"], default="single")
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--repeats", type=int, default=50)
    bench.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser(
        "summarize", help="Recalculate summary from raw samples"
    )
    summary.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            print(
                json.dumps(
                    probe_collectives(
                        payload_bytes=args.payload_bytes,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                ),
                flush=True,
            )
            return 0
        if args.command == "summarize":
            print(json.dumps(summarize_directory(args.input), indent=2))
            return 0
        device = mx.gpu if args.device == "gpu" else mx.cpu
        if args.command == "check":
            config = config_from(args, args.length)
            q, k, v = make_inputs(config)
            with mx.stream(device):
                reference = reference_attention(q, k, v)
                errors = {
                    "optimized": error_metrics(
                        optimized_attention(q, k, v), reference, args.dtype
                    ),
                    "local-context": error_metrics(
                        context_attention(q, k, v, block_size=args.block_size),
                        reference,
                        args.dtype,
                    ),
                }
            print(json.dumps(errors, indent=2))
            return 0 if all(error["passed"] for error in errors.values()) else 1
        configs = [config_from(args, length) for length in args.lengths]
        if len(set(args.lengths)) != len(args.lengths):
            raise ValueError("Lengths must be unique")
        if args.warmup < 0 or args.repeats <= 0:
            raise ValueError("Warmup must be nonnegative and repeats must be positive")
        args.output.mkdir(parents=True, exist_ok=False)
        for config in configs:
            print(
                json.dumps(
                    benchmark_local(
                        config,
                        mode=args.mode,
                        output=args.output / f"{args.mode}-{config.length}",
                        device=device,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                ),
                flush=True,
            )
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"weftlm: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
