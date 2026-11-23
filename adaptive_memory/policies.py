"""Deterministic conventional policies and a causal online reuse predictor.

All policy timestamps refer to requests reaching that cache, not CPU instructions.
Learned-policy labels mature H LLC references after the feature snapshot. The
current request is observed before labels due now are settled. No future trace
is passed to a policy. Incomplete windows at EOF are censored, never negatives.
"""

import math
import random
from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(slots=True)
class Line:
    block: int
    inserted: int
    last: int
    count: int = 1
    rrpv: int = 2
    gap: int = 0
    frequency: float = 1.0


class Policy:
    def before(self, block, lines, tick):
        pass

    def hit(self, line, tick):
        line.last = tick
        line.count += 1

    def inserted(self, line, tick):
        pass

    def victim(self, lines, tick):
        raise NotImplementedError

    def diagnostics(self):
        return {}


class LRU(Policy):
    def victim(self, lines, tick):
        return min(lines, key=lambda x: (x.last, x.block))


class FIFO(Policy):
    def victim(self, lines, tick):
        return min(lines, key=lambda x: (x.inserted, x.block))


class LFU(Policy):
    """Resident lifetime frequency; counts reset when a line is evicted."""

    def victim(self, lines, tick):
        return min(lines, key=lambda x: (x.count, x.last, x.block))


class SRRIP(Policy):
    """Two-bit SRRIP-HP: insert at 2, hit at 0, age until a 3 exists."""

    def hit(self, line, tick):
        super().hit(line, tick)
        line.rrpv = 0

    def victim(self, lines, tick):
        while True:
            for line in lines:
                if line.rrpv == 3:
                    return line
            for line in lines:
                line.rrpv = min(3, line.rrpv + 1)


@dataclass(slots=True)
class Sample:
    block: int
    due: int
    features: tuple
    probability: float
    reused: bool = False


class OnlineReuse(LRU):
    """Sampled logistic reuse predictor; research prototype, not hardware logic.

    A single uniformly sampled resident of the requested set is snapshotted
    before every LLC lookup. At most H samples and O(H) target IDs are live.
    Features describe the same candidate state at training and eviction time.
    Warmup trains all ML variants; frozen disables updates after warmup.
    """

    def __init__(self, config, seed=0, frozen=False, recency_only=False):
        self.horizon = config.horizon
        self.rate = config.learning_rate
        self.min_labels = config.min_labels
        self.rng = random.Random(seed)
        self.frozen = frozen
        self.recency_only = recency_only
        self.training = True
        self.weights = [0.0] * (2 if recency_only else 6)
        self.pending = deque()
        self.by_block = {}
        self.labels = self.positives = self.updates = 0
        self.brier_sum = self.logloss_sum = 0.0
        self.prediction_calls = self.prediction_ns = self.update_ns = 0
        self.sample_peak = self.target_peak = 0
        self.fallback_evictions = self.learned_evictions = 0

    def start_measurement(self):
        if self.frozen:
            self.training = False

    def features(self, line, tick):
        age = max(0, tick - line.last)
        normalizer = math.log1p(self.horizon)
        recency = min(2.0, math.log1p(age) / normalizer)
        if self.recency_only:
            return (1.0, recency)
        decayed = line.frequency * math.exp(-age / self.horizon)
        gap = min(2.0, math.log1p(line.gap) / normalizer)
        overdue = min(2.0, age / max(1, line.gap)) / 2.0
        return (1.0, recency, math.log1p(decayed) / normalizer,
                gap, overdue, float(line.count > 1))

    def predict(self, features):
        start = perf_counter_ns()
        logit = sum(w * x for w, x in zip(self.weights, features))
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        self.prediction_ns += perf_counter_ns() - start
        self.prediction_calls += 1
        return probability

    def settle(self, sample):
        y = int(sample.reused)
        p = max(1e-12, min(1 - 1e-12, sample.probability))
        self.labels += 1
        self.positives += y
        # Prequential metrics use the prediction saved at snapshot time.
        self.brier_sum += (p - y) ** 2
        self.logloss_sum -= y * math.log(p) + (1 - y) * math.log(1 - p)
        if self.training:
            start = perf_counter_ns()
            logit = sum(w * x for w, x in zip(self.weights, sample.features))
            current = 1 / (1 + math.exp(-max(-30, min(30, logit))))
            error = current - y
            for i, x in enumerate(sample.features):
                penalty = 0.0001 * self.weights[i] if i else 0.0
                self.weights[i] -= self.rate * (error * x + penalty)
            self.update_ns += perf_counter_ns() - start
            self.updates += 1

    def before(self, block, lines, tick):
        # An access at exactly the endpoint belongs to (snapshot, snapshot+H].
        for sample in self.by_block.get(block, ()):
            sample.reused = True
        while self.pending and self.pending[0].due <= tick:
            sample = self.pending.popleft()
            targets = self.by_block[sample.block]
            assert targets[0] is sample
            targets.popleft()
            if not targets:
                del self.by_block[sample.block]
            self.settle(sample)
        if lines:
            candidate = self.rng.choice(lines)
            features = self.features(candidate, tick)
            sample = Sample(candidate.block, tick + self.horizon,
                            features, self.predict(features))
            self.pending.append(sample)
            self.by_block.setdefault(candidate.block, deque()).append(sample)
            self.sample_peak = max(self.sample_peak, len(self.pending))
            self.target_peak = max(self.target_peak, len(self.by_block))

    def hit(self, line, tick):
        gap = tick - line.last
        line.frequency = line.frequency * math.exp(-gap / self.horizon) + 1
        line.gap = gap
        super().hit(line, tick)

    def victim(self, lines, tick):
        if self.updates < self.min_labels:
            self.fallback_evictions += 1
            return super().victim(lines, tick)
        self.learned_evictions += 1
        return min(lines, key=lambda x: (self.predict(self.features(x, tick)), x.last, x.block))

    def diagnostics(self):
        return {
            "labels": self.labels, "positive_labels": self.positives,
            "updates": self.updates, "pending_censored": len(self.pending),
            "sample_peak": self.sample_peak, "target_peak": self.target_peak,
            "prediction_calls": self.prediction_calls,
            "prediction_ns": self.prediction_ns, "update_ns": self.update_ns,
            "brier": self.brier_sum / self.labels if self.labels else None,
            "logloss": self.logloss_sum / self.labels if self.labels else None,
            "weights": self.weights.copy(),
            "fallback_evictions": self.fallback_evictions,
            "learned_evictions": self.learned_evictions,
        }


POLICIES = ("lru", "fifo", "lfu", "srrip", "online", "frozen", "recency")


def make_policy(name, config, seed=0):
    if name in {"online", "frozen", "recency"}:
        return OnlineReuse(config, seed, frozen=name == "frozen", recency_only=name == "recency")
    constructors = {"lru": LRU, "fifo": FIFO, "lfu": LFU, "srrip": SRRIP}
    if name not in constructors:
        raise ValueError(f"Unknown policy {name!r}; choose from {POLICIES}")
    return constructors[name]()

