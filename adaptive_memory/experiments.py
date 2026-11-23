"""Paired trace experiments; statistics over independent generator seeds."""

import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .simulator import simulate
from .workloads import generate

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_ci(values, seed=2026, repeats=2000):
    """Percentile paired-seed bootstrap. Not a population-wide guarantee."""
    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [means[int(0.025 * repeats)], means[min(repeats - 1, int(0.975 * repeats))]]


def summarize(runs):
    groups = defaultdict(list)
    baseline = {(r["workload"], r["config"]["l3_lines"], r["seed"]): r
                for r in runs if r["policy"] == "lru"}
    for run in runs:
        groups[(run["workload"], run["config"]["l3_lines"], run["policy"])].append(run)
    summary = []
    for (workload, size, policy), group in sorted(groups.items()):
        row = {"workload": workload, "l3_lines": size, "policy": policy, "runs": len(group)}
        for metric in ("amat_cycles", "ram_miss_ratio", "cache_hit_ratio", "fill_bytes", "runtime_ms"):
            values = [r[metric] for r in group]
            row[metric] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        paired = []
        for run in group:
            base = baseline.get((workload, size, run["seed"]))
            if base:
                if base["trace_sha256"] != run["trace_sha256"]:
                    raise ValueError("Paired policies must use identical trace hashes")
                paired.append(100 * (base["amat_cycles"] - run["amat_cycles"]) / base["amat_cycles"])
        row["amat_reduction_vs_lru_pct"] = statistics.mean(paired) if paired else None
        row["amat_reduction_ci95"] = bootstrap_ci(paired) if paired else None
        summary.append(row)
    return summary


def source_hashes():
    paths = list((ROOT / "adaptive_memory").glob("*.py"))
    paths += list((ROOT / "configs").glob("*.json"))
    paths += list((ROOT / "tests").glob("*.py"))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def write_results(runs, manifest, directory):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    payload = {"manifest": manifest, "summary": summary, "runs": runs}
    (output / "benchmark.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    flat_fields = ["workload", "l3_lines", "policy", "seed", "trace_sha256", "warmup",
                   "measured_requests", "amat_cycles", "ram_miss_ratio", "cache_hit_ratio",
                   "fill_bytes", "runtime_ms", "prediction_ms", "update_ms", "labels", "updates"]
    with (output / "raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, flat_fields)
        writer.writeheader()
        for run in runs:
            row = {key: run[key] for key in flat_fields if key in run}
            row["l3_lines"] = run["config"]["l3_lines"]
            diag = run["diagnostics"]
            row["prediction_ms"] = diag.get("measured_prediction_ns", 0) / 1e6
            row["update_ms"] = diag.get("measured_update_ns", 0) / 1e6
            row["labels"] = diag.get("measured_labels", 0)
            row["updates"] = diag.get("measured_updates", 0)
            writer.writerow(row)
    if summary:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return payload


def benchmark(spec, directory="results", progress=print):
    measured = spec["measured_requests"]
    warmup = spec["warmup"]
    if measured <= 0 or warmup < 0:
        raise ValueError("Experiment needs positive measurement length and nonnegative warmup")
    for key in ("seeds", "l3_lines", "workloads", "policies"):
        if not spec[key] or len(spec[key]) != len(set(spec[key])):
            raise ValueError(f"{key} must be nonempty and contain unique values")
    runs = []
    total = len(spec["workloads"]) * len(spec["l3_lines"]) * len(spec["seeds"]) * len(spec["policies"])
    progress(f"Running {total} simulations; {warmup:,} warmup + {measured:,} measured requests each")
    start = datetime.now(timezone.utc).isoformat()
    hashes_at_start = source_hashes()
    for workload in spec["workloads"]:
        for seed in spec["seeds"]:
            addresses = generate(workload, measured + warmup, seed, spec.get("simulator", {}).get("line_bytes", 64))
            for size in spec["l3_lines"]:
                config_values = dict(spec.get("simulator", {}))
                config_values["l3_lines"] = size
                config = Config(**config_values)
                policies = list(spec["policies"])
                # Fixed, reproducible shuffled order mitigates systematic timing-order bias.
                random.Random(f"{workload}:{seed}:{size}").shuffle(policies)
                for policy in policies:
                    result = simulate(addresses, config, policy, warmup, seed)
                    result["workload"] = workload
                    runs.append(result)
                progress(f"[{len(runs):3}/{total}] {workload:12} seed={seed:2} L3={size:4}")
    manifest = {
        "study": spec, "started_utc": start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "source_sha256": hashes_at_start,
        "source_unchanged_during_run": hashes_at_start == source_hashes(),
        "run_count": len(runs), "trace_type": "synthetic read-only byte-address traces",
        "model": "serial non-inclusive functional caches; no CPU execution or hardware overhead model",
        "uncertainty": "95% percentile bootstrap of paired generator seeds within each workload and size; five seeds are exploratory",
    }
    payload = write_results(runs, manifest, directory)
    progress(f"Saved {len(runs)} runs in {Path(directory).resolve()}")
    return payload

