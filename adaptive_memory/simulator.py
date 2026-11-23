"""Serial functional model: read-only, non-inclusive, no timing overlap."""

import hashlib
import struct
from dataclasses import dataclass
from time import perf_counter_ns

from .config import Config
from .policies import LRU, Line, make_policy


class Cache:
    def __init__(self, lines, ways, policy):
        if lines <= 0 or ways <= 0 or lines % ways:
            raise ValueError("Cache lines must be positive and divisible by ways")
        self.ways = ways
        self.sets = [dict() for _ in range(lines // ways)]
        self.policy = policy
        self.tick = 0
        self.hits = self.misses = self.fills = self.evictions = 0

    def access(self, block):
        self.tick += 1
        entries = self.sets[block % len(self.sets)]
        self.policy.before(block, list(entries.values()), self.tick)
        if block in entries:
            self.hits += 1
            self.policy.hit(entries[block], self.tick)
            return True
        self.misses += 1
        if len(entries) == self.ways:
            victim = self.policy.victim(list(entries.values()), self.tick)
            del entries[victim.block]
            self.evictions += 1
        line = Line(block, self.tick, self.tick)
        self.policy.inserted(line, self.tick)
        entries[block] = line
        self.fills += 1
        return False

    def counts(self):
        return {key: getattr(self, key) for key in ("hits", "misses", "fills", "evictions")}


@dataclass
class Access:
    served_by: str
    cycles: int
    fill_bytes: int


class Hierarchy:
    def __init__(self, config=None, policy="online", seed=0):
        self.config = config or Config()
        c = self.config
        self.l1 = Cache(c.l1_lines, c.l1_ways, LRU())
        self.l2 = Cache(c.l2_lines, c.l2_ways, LRU())
        self.l3 = Cache(c.l3_lines, c.l3_ways, make_policy(policy, c, seed))

    def access(self, address):
        if isinstance(address, bool) or not isinstance(address, int) or not 0 <= address < 2 ** 64:
            raise ValueError("Addresses must be unsigned 64-bit integer byte addresses")
        c = self.config
        block = address // c.line_bytes
        if self.l1.access(block):
            return Access("L1", c.l1_cycles, 0)
        if self.l2.access(block):
            return Access("L2", c.l1_cycles + c.l2_cycles, c.line_bytes)
        if self.l3.access(block):
            return Access("L3", c.l1_cycles + c.l2_cycles + c.l3_cycles, 2 * c.line_bytes)
        return Access("RAM", c.l1_cycles + c.l2_cycles + c.l3_cycles + c.ram_cycles,
                      3 * c.line_bytes)


def trace_hash(addresses):
    digest = hashlib.sha256()
    for address in addresses:
        digest.update(struct.pack("<Q", address))
    return digest.hexdigest()


def simulate(addresses, config=None, policy="online", warmup=0, seed=0):
    c = config or Config()
    if not addresses:
        raise ValueError("Trace must contain at least one address")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < len(addresses):
        raise ValueError("warmup must be an integer smaller than trace length")
    hierarchy = Hierarchy(c, policy, seed)
    start = perf_counter_ns()
    for address in addresses[:warmup]:
        hierarchy.access(address)
    warmup_ns = perf_counter_ns() - start
    learned = hierarchy.l3.policy
    if hasattr(learned, "start_measurement"):
        learned.start_measurement()
    before = [cache.counts() for cache in (hierarchy.l1, hierarchy.l2, hierarchy.l3)]
    diag_before = learned.diagnostics()
    served = {"L1": 0, "L2": 0, "L3": 0, "RAM": 0}
    cycles = fill_bytes = 0
    windows = []
    window_cycles = window_ram = 0
    start = perf_counter_ns()
    for index, address in enumerate(addresses[warmup:], 1):
        access = hierarchy.access(address)
        served[access.served_by] += 1
        cycles += access.cycles
        fill_bytes += access.fill_bytes
        window_cycles += access.cycles
        window_ram += access.served_by == "RAM"
        if index % c.window == 0 or index == len(addresses) - warmup:
            count = (index - 1) % c.window + 1
            windows.append({"end": index, "requests": count,
                            "amat_cycles": window_cycles / count,
                            "ram_misses": window_ram})
            window_cycles = window_ram = 0
    runtime_ns = perf_counter_ns() - start
    measured = len(addresses) - warmup
    levels = {}
    for name, cache, old in zip(("L1", "L2", "L3"),
                                (hierarchy.l1, hierarchy.l2, hierarchy.l3), before):
        counts = {key: value - old[key] for key, value in cache.counts().items()}
        requests = counts["hits"] + counts["misses"]
        counts["local_hit_ratio"] = counts["hits"] / requests if requests else None
        levels[name] = counts
    diagnostics = learned.diagnostics()
    # Whole-run prediction quality is explicitly separate from measured-phase timers.
    for key in ("prediction_ns", "update_ns", "prediction_calls", "updates", "labels"):
        if key in diagnostics:
            diagnostics[f"measured_{key}"] = diagnostics[key] - diag_before[key]
    return {
        "policy": policy, "seed": seed, "config": c.to_dict(),
        "trace_sha256": trace_hash(addresses), "total_requests": len(addresses),
        "warmup": warmup, "measured_requests": measured,
        "served": served, "levels": levels, "total_cycles": cycles,
        "amat_cycles": cycles / measured,
        "ram_miss_ratio": served["RAM"] / measured,
        "cache_hit_ratio": 1 - served["RAM"] / measured,
        "llc_misses_per_1000_requests": levels["L3"]["misses"] * 1000 / measured,
        "fill_bytes": fill_bytes,
        "runtime_ms": runtime_ns / 1e6, "warmup_runtime_ms": warmup_ns / 1e6,
        "diagnostics": diagnostics, "windows": windows,
    }

