"""Validate experiment accounting, paired comparisons, and source provenance."""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify():
    path = ROOT / "results/benchmark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    study = data["manifest"]["study"]
    expected = len(study["workloads"]) * len(study["l3_lines"]) * len(study["policies"]) * len(study["seeds"])
    assert len(data["runs"]) == expected, "Incomplete matrix"
    seen, paired = set(), {}
    for run in data["runs"]:
        key = (run["workload"], run["config"]["l3_lines"], run["policy"], run["seed"])
        assert key not in seen, "Duplicate simulation"
        seen.add(key)
        assert key[0] in study["workloads"] and key[1] in study["l3_lines"]
        assert key[2] in study["policies"] and key[3] in study["seeds"]
        n = run["measured_requests"]
        assert n == study["measured_requests"]
        assert run["warmup"] == study["warmup"]
        served, levels, config = run["served"], run["levels"], run["config"]
        assert sum(served.values()) == n
        assert levels["L1"]["hits"] == served["L1"]
        assert levels["L1"]["misses"] == served["L2"] + served["L3"] + served["RAM"]
        assert levels["L2"]["hits"] == served["L2"]
        assert levels["L2"]["misses"] == served["L3"] + served["RAM"]
        assert levels["L3"]["hits"] == served["L3"]
        assert levels["L3"]["misses"] == served["RAM"]
        cycles = config["l1_cycles"] * n + config["l2_cycles"] * levels["L1"]["misses"]
        cycles += config["l3_cycles"] * levels["L2"]["misses"] + config["ram_cycles"] * levels["L3"]["misses"]
        assert cycles == run["total_cycles"]
        assert math.isclose(cycles / n, run["amat_cycles"])
        assert run["fill_bytes"] == config["line_bytes"] * sum(levels[k]["fills"] for k in ("L1", "L2", "L3"))
        assert sum(w["requests"] for w in run["windows"]) == n
        assert sum(w["ram_misses"] for w in run["windows"]) == served["RAM"]
        assert math.isclose(sum(w["requests"] * w["amat_cycles"] for w in run["windows"]), cycles)
        pair_key = (key[0], key[1], key[3])
        fingerprint = (run["trace_sha256"], levels["L1"]["misses"], levels["L2"]["misses"])
        assert paired.setdefault(pair_key, fingerprint) == fingerprint, "Unfair upper-cache or trace pairing"
        diagnostics = run["diagnostics"]
        if diagnostics:
            assert diagnostics["sample_peak"] <= config["horizon"]
            assert diagnostics["target_peak"] <= config["horizon"]
            assert diagnostics["pending_censored"] <= config["horizon"]
            assert all(math.isfinite(w) for w in diagnostics["weights"])
            if run["policy"] == "frozen":
                assert diagnostics["measured_updates"] == 0
    assert data["manifest"]["source_unchanged_during_run"], "Source changed during benchmark"
    for relative, digest in data["manifest"]["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, f"Source drift: {relative}"
    from adaptive_memory.experiments import summarize
    assert summarize(data["runs"]) == data["summary"], "Stored summary differs from recomputation"
    provenance_path = ROOT / "paper/provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert provenance["results_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), "Paper uses other results"
    print(f"PASS: {len(seen)} simulations, paired traces, accounting, bounded metadata, source hashes, summaries, and paper provenance")
    return {"simulations": len(seen), "paired_cases": len(paired), "status": "passed"}


if __name__ == "__main__":
    verify()

