import argparse
import json
from pathlib import Path

from .config import Config
from .experiments import ROOT, benchmark
from .policies import POLICIES
from .simulator import simulate
from .workloads import WORKLOADS, generate, read_trace


def main():
    parser = argparse.ArgumentParser(prog="python -m adaptive_memory", description="Adaptive Memory Lab")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("simulate", help="Run one policy on a synthetic or CSV trace")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--workload", choices=WORKLOADS, default="phase_shift")
    source.add_argument("--trace", type=Path)
    run.add_argument("--policy", choices=POLICIES, default="online")
    run.add_argument("--requests", type=int, default=15000, help="Total requests including warmup")
    run.add_argument("--warmup", type=int, default=3000)
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--l3-lines", type=int, default=512)
    run.add_argument("--horizon", type=int, default=256)
    run.add_argument("--output", type=Path)
    suite = commands.add_parser("benchmark", help="Run a reproducible experiment matrix")
    suite.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    suite.add_argument("--output", type=Path, default=ROOT / "results")
    report = commands.add_parser("report", help="Generate figures and research paper from existing results")
    report.add_argument("--results", type=Path, default=ROOT / "results/benchmark.json")
    server = commands.add_parser("serve", help="Open the local interactive experiment dashboard")
    server.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == "simulate":
            config = Config(l3_lines=args.l3_lines, horizon=args.horizon)
            addresses = read_trace(args.trace) if args.trace else generate(args.workload, args.requests, args.seed)
            result = simulate(addresses, config, args.policy, args.warmup, args.seed)
            result["workload"] = str(args.trace) if args.trace else args.workload
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            display = {key: result[key] for key in ("policy", "workload", "measured_requests", "served", "amat_cycles", "cache_hit_ratio", "fill_bytes", "runtime_ms")}
            print(json.dumps(display, indent=2))
        elif args.command == "benchmark":
            benchmark(json.loads(args.config.read_text(encoding="utf-8")), args.output)
        elif args.command == "report":
            from .reporting import build_report
            build_report(args.results)
        elif args.command == "serve":
            from .server import serve
            serve(args.port)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

