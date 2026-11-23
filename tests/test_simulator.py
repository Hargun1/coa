import csv
import random

import pytest

from adaptive_memory.config import Config
from adaptive_memory.experiments import bootstrap_ci, summarize
from adaptive_memory.policies import FIFO, LFU, LRU, SRRIP, OnlineReuse
from adaptive_memory.simulator import Cache, Hierarchy, simulate
from adaptive_memory.workloads import WORKLOADS, generate, read_trace


def tiny(**kwargs):
    return Config(l1_lines=1, l1_ways=1, l2_lines=2, l2_ways=1,
                  l3_lines=4, l3_ways=2, **kwargs)


def test_hand_calculated_hierarchy_and_transfer_cost():
    h = Hierarchy(tiny(), "lru")
    # 0 RAM, 0 L1, 1 RAM, 0 L2, 2 RAM, 0 L3.
    result = [h.access(b * 64) for b in [0, 0, 1, 0, 2, 0]]
    assert [r.served_by for r in result] == ["RAM", "L1", "RAM", "L2", "RAM", "L3"]
    assert [r.cycles for r in result] == [256, 4, 256, 16, 256, 56]
    assert [r.fill_bytes for r in result] == [192, 0, 192, 64, 192, 128]


@pytest.mark.parametrize("policy,sequence,remaining", [
    (LRU, [0, 1, 0, 2], {0, 2}),
    (FIFO, [0, 1, 0, 2], {1, 2}),
    (LFU, [0, 1, 0, 2], {0, 2}),
    (SRRIP, [0, 1, 0, 2], {0, 2}),
])
def test_replacement_decisions(policy, sequence, remaining):
    cache = Cache(2, 2, policy())
    for block in sequence:
        cache.access(block)
    assert set(cache.sets[0]) == remaining


def test_line_offsets_and_conflicting_sets():
    h = Hierarchy(tiny(), "lru")
    assert h.access(0).served_by == "RAM"
    assert h.access(63).served_by == "L1"
    assert h.access(64).served_by == "RAM"
    cache = Cache(4, 2, LRU())
    for block in (0, 2, 4):
        cache.access(block)
    assert set(cache.sets[0]) == {2, 4}
    assert not cache.sets[1]


def test_warmup_excluded_from_counters_but_keeps_cache_state():
    result = simulate([0, 0, 0], tiny(), "lru", warmup=1)
    assert result["served"] == {"L1": 2, "L2": 0, "L3": 0, "RAM": 0}
    assert result["amat_cycles"] == 4
    assert result["levels"]["L3"]["local_hit_ratio"] is None


def test_delayed_labels_include_endpoint_and_never_current_snapshot():
    p = OnlineReuse(tiny(horizon=2, min_labels=1))
    cache = Cache(1, 1, p)
    cache.access(0)  # t1 no resident to sample
    cache.access(1)  # t2 snapshot block0; current block1 cannot label it
    cache.access(2)  # t3 no matured label
    assert p.labels == p.updates == 0
    cache.access(0)  # t4 block0 at exact endpoint is positive
    assert p.labels == 1
    assert p.positives == 1
    assert p.updates == 1


def test_no_reuse_produces_negative_only_after_full_horizon():
    p = OnlineReuse(tiny(horizon=2))
    cache = Cache(1, 1, p)
    for block in range(4):
        cache.access(block)
    assert p.labels == 1 and p.positives == 0
    assert p.diagnostics()["pending_censored"] == 2


def test_future_suffix_cannot_change_prefix_state():
    addresses = generate("random", 250, 17)
    left, right = Hierarchy(tiny(horizon=8), "online", 7), Hierarchy(tiny(horizon=8), "online", 7)
    for address in addresses:
        assert left.access(address).served_by == right.access(address).served_by
    assert left.l3.policy.weights == right.l3.policy.weights
    assert [set(s) for s in left.l3.sets] == [set(s) for s in right.l3.sets]
    # Only one hierarchy sees the arbitrary unseen suffix.
    prefix_weights = left.l3.policy.weights.copy()
    for address in generate("looping", 100, 99):
        right.access(address)
    assert left.l3.policy.weights == prefix_weights


def test_model_and_target_metadata_are_bounded():
    p = OnlineReuse(tiny(horizon=7))
    cache = Cache(2, 2, p)
    for block in range(1000):
        cache.access(block)
    assert p.sample_peak <= 7
    assert p.target_peak <= 7
    assert len(p.pending) <= 7
    assert len(p.weights) == 6


def test_frozen_stops_weight_updates_after_warmup():
    p = OnlineReuse(tiny(horizon=2, min_labels=1), frozen=True)
    cache = Cache(2, 2, p)
    for i in range(40):
        cache.access(i % 4)
    weights, updates = p.weights.copy(), p.updates
    p.start_measurement()
    for i in range(40):
        cache.access(i + 100)
    assert p.updates == updates and p.weights == weights
    assert p.labels > updates


@pytest.mark.parametrize("workload", WORKLOADS)
def test_policy_pairing_and_conservation(workload):
    trace = generate(workload, 350, seed=23)
    results = [simulate(trace, tiny(horizon=8), policy, warmup=50, seed=23)
               for policy in ("lru", "fifo", "lfu", "srrip", "online", "frozen", "recency")]
    assert len({r["trace_sha256"] for r in results}) == 1
    assert len({r["levels"]["L1"]["misses"] for r in results}) == 1
    assert len({r["levels"]["L2"]["misses"] for r in results}) == 1
    for r in results:
        assert sum(r["served"].values()) == 300
        assert r["levels"]["L1"]["misses"] == sum(r["served"][k] for k in ("L2", "L3", "RAM"))
        assert r["levels"]["L2"]["misses"] == r["served"]["L3"] + r["served"]["RAM"]
        assert r["levels"]["L3"]["misses"] == r["served"]["RAM"]
        assert r["total_cycles"] == sum(r["served"][k] * cost for k, cost in zip(("L1", "L2", "L3", "RAM"), (4, 16, 56, 256)))
        assert sum(w["requests"] for w in r["windows"]) == 300
        assert r["fill_bytes"] == 64 * sum(r["levels"][k]["fills"] for k in ("L1", "L2", "L3"))


def test_bootstrap_and_summary_pairing():
    assert bootstrap_ci([3, 3, 3]) == [3, 3]
    trace = generate("random", 100, 1)
    runs = []
    for policy in ("lru", "online"):
        result = simulate(trace, tiny(horizon=8), policy, seed=1)
        result["workload"] = "random"
        runs.append(result)
    assert len(summarize(runs)) == 2
    runs[1]["trace_sha256"] = "corrupted"
    with pytest.raises(ValueError, match="trace hashes"):
        summarize(runs)


def test_strict_trace_import(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("address,operation\n0x0,R\n64,READ\n", encoding="utf-8")
    assert read_trace(path) == [0, 64]
    path.write_text("address,operation\n0,W\n", encoding="utf-8")
    with pytest.raises(ValueError, match="read operations"):
        read_trace(path)
    path.write_text("-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="64-bit"):
        read_trace(path)
    path.write_text("1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="limit"):
        read_trace(path, limit=1)


@pytest.mark.parametrize("kwargs", [{"l3_lines": 513}, {"line_bytes": 63}, {"horizon": 0},
                                   {"learning_rate": -1}, {"window": True}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("trace,warmup", [([], 0), ([0], 1), ([0], -1), ([0], 0.5)])
def test_invalid_simulation(trace, warmup):
    with pytest.raises(ValueError):
        simulate(trace, warmup=warmup)


@pytest.mark.parametrize("address", [-1, True, 1.25, 2 ** 64])
def test_invalid_address(address):
    with pytest.raises(ValueError):
        Hierarchy().access(address)
