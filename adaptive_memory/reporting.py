"""Generate publication figures, editable manuscript sources, and a reviewed PDF.

Every numerical claim is derived from benchmark.json. This is a student study
of a functional simulator; neither novelty nor physical CPU speedup is claimed.
"""

import hashlib
import json
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

from .experiments import ROOT

LABELS = {"sequential": "Sequential", "looping": "Looping", "random": "Uniform random",
          "zipf": "Zipf popularity", "phase_shift": "Changing hotspots", "mixed_scan": "Hotset + scans"}
POLICY_LABELS = {"lru": "LRU", "fifo": "FIFO", "lfu": "LFU", "srrip": "SRRIP",
                 "online": "Online", "frozen": "Frozen", "recency": "Recency only"}
COLORS = {"lru": "#82969f", "srrip": "#c17a37", "online": "#176b53",
          "frozen": "#354c67", "recency": "#8a8056", "fifo": "#a5a5a5", "lfu": "#637d56"}

REFERENCES = [
    ("rrip", "A. Jaleel, K. B. Theobald, S. C. Steely Jr., and J. Emer, 'High Performance Cache Replacement Using Re-Reference Interval Prediction (RRIP),' ISCA, pp. 60-71, 2010. DOI: 10.1145/1815961.1815971.", "https://jaleels.org/ajaleel/publications/isca2010-rrip.pdf"),
    ("hawkeye", "A. Jain and C. Lin, 'Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement,' ISCA, pp. 78-89, 2016. DOI: 10.1109/ISCA.2016.17.", "https://class.ece.iastate.edu/tyagi/cpre581/papers/ISCA16Belady.pdf"),
    ("lecar", "G. Vietri et al., 'Driving Cache Replacement with ML-based LeCaR,' USENIX HotStorage, 2018.", "https://www.usenix.org/conference/hotstorage18/presentation/vietri"),
    ("lrb", "Z. Song, D. S. Berger, K. Li, and W. Lloyd, 'Learning Relaxed Belady for Content Distribution Network Caching,' USENIX NSDI, pp. 529-544, 2020.", "https://www.cs.princeton.edu/~wlloyd/papers/lrb-nsdi20.pdf"),
    ("artmem", "X. Yi, H. Du, Y. Wang, J. Zhang, Q. Li, and C. J. Xue, 'ArtMem: Adaptive Migration in Reinforcement Learning-Enabled Tiered Memory,' ISCA, pp. 405-418, 2025. DOI: 10.1145/3695053.3731001.", "https://doi.org/10.1145/3695053.3731001"),
    ("champsim", "ChampSim contributors, 'ChampSim documentation,' online project documentation. Accessed November 20, 2026.", "https://champsim.github.io/ChampSim/develop/"),
]


def lookup(payload, workload, size, policy):
    return next(r for r in payload["summary"] if r["workload"] == workload and r["l3_lines"] == size and r["policy"] == policy)


def figures(payload, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#162b34", "text.color": "#162b34",
                         "axes.titleweight": "semibold", "savefig.facecolor": "white"})
    workloads = payload["manifest"]["study"]["workloads"]
    sizes = payload["manifest"]["study"]["l3_lines"]
    # Figure 1: all workload/capacity cases, including ties and regressions.
    matrix = np.array([[lookup(payload, w, s, "online")["amat_reduction_vs_lru_pct"] for s in sizes] for w in workloads])
    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout="constrained")
    image = ax.imshow(matrix, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=min(-8, float(matrix.min())), vcenter=0, vmax=max(30, float(matrix.max()))), aspect="auto")
    ax.set_xticks(range(len(sizes)), [f"{s * 64 // 1024} KiB ({s} lines)" for s in sizes])
    ax.set_yticks(range(len(workloads)), [LABELS[w] for w in workloads])
    for i in range(len(workloads)):
        for j in range(len(sizes)):
            ax.text(j, i, f"{matrix[i,j]:+.2f}%", ha="center", va="center", color="#162b34", fontsize=10)
    ax.set_xlabel("L3 capacity; positive values mean lower modeled AMAT than LRU")
    fig.colorbar(image, ax=ax, label="AMAT reduction (%)", shrink=0.85)
    fig.savefig(directory / "reduction.png", dpi=200)
    plt.close(fig)
    # Figure 2: absolute cycle costs prevent relative-only interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, layout="constrained")
    selected = ["lru", "srrip", "online", "frozen"]
    for ax, size in zip(axes, sizes[:2]):
        x = np.arange(len(workloads))
        for i, policy in enumerate(selected):
            values = [lookup(payload, w, size, policy)["amat_cycles"] for w in workloads]
            ax.bar(x + (i - 1.5) * 0.19, values, width=0.18, label=POLICY_LABELS[policy], color=COLORS[policy])
        ax.set_xticks(x, [LABELS[w] for w in workloads], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"L3 = {size * 64 // 1024} KiB")
        ax.set_ylim(0, 280)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Modeled AMAT (cycles / CPU memory request)")
    axes[1].legend(ncol=2, fontsize=8, loc="upper right")
    fig.savefig(directory / "absolute_amat.png", dpi=200)
    plt.close(fig)
    # Figure 3: direct average over the same request windows and seeds.
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.7), layout="constrained")
    for ax, (workload, size) in zip(axes, [("phase_shift", 512), ("mixed_scan", 256)]):
        for policy in selected:
            runs = [r for r in payload["runs"] if r["workload"] == workload and r["policy"] == policy and r["config"]["l3_lines"] == size]
            windows = runs[0]["windows"]
            ys = [statistics.mean(r["windows"][i]["amat_cycles"] for r in runs) for i in range(len(windows))]
            ax.plot([w["end"] for w in windows], ys, marker=".", ms=4, lw=1.4, label=POLICY_LABELS[policy], color=COLORS[policy])
        if workload == "phase_shift":
            study = payload["manifest"]["study"]
            for fraction in (0.25, 0.5, 0.75):
                boundary = fraction * (study["warmup"] + study["measured_requests"]) - study["warmup"]
                ax.axvline(boundary, color="#aeb9b1", ls=":", lw=0.9)
        ax.set_title(f"{LABELS[workload]}, L3 = {size * 64 // 1024} KiB", fontsize=10)
        ax.set_ylabel("AMAT (cycles)")
        ax.grid(alpha=0.15)
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("Measured CPU memory requests after warmup (1,000-request windows)")
    fig.savefig(directory / "adaptation.png", dpi=200)
    plt.close(fig)
    # Figure 4: actual Python runtime; no conversion into modeled CPU cycles.
    fig, ax = plt.subplots(figsize=(7.6, 2.8), layout="constrained")
    x = np.arange(len(workloads))
    for i, policy in enumerate(("online", "frozen", "recency")):
        ratios = [lookup(payload, w, 512, policy)["runtime_ms"] / lookup(payload, w, 512, "lru")["runtime_ms"] for w in workloads]
        ax.bar(x + (i - 1) * 0.23, ratios, width=0.22, color=COLORS[policy], label=POLICY_LABELS[policy])
    ax.axhline(1, color="#82969f", ls="--", lw=1)
    ax.set_xticks(x, [LABELS[w] for w in workloads], rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Python runtime / LRU runtime")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.savefig(directory / "runtime.png", dpi=200)
    plt.close(fig)


def manuscript(payload):
    study = payload["manifest"]["study"]
    online = [r for r in payload["summary"] if r["policy"] == "online"]
    gains = [r["amat_reduction_vs_lru_pct"] for r in online]
    average = statistics.mean(gains)
    best, worst = max(gains), min(gains)
    wins = sum(g > 1e-9 for g in gains)
    ties = sum(abs(g) <= 1e-9 for g in gains)
    losses = len(gains) - wins - ties
    runtimes = [r["runtime_ms"] / lookup(payload, r["workload"], r["l3_lines"], "lru")["runtime_ms"] for r in online]
    phase = {p: lookup(payload, "phase_shift", 512, p)["amat_cycles"] for p in ("lru", "online", "frozen", "recency")}
    mixed = {p: lookup(payload, "mixed_scan", 256, p)["amat_cycles"] for p in ("lru", "online", "frozen", "srrip")}
    # Structured blocks feed PDF, Markdown, and LaTeX from the same content.
    sections = []
    def section(title, *blocks):
        sections.append((title, list(blocks)))
    def p(text):
        return ("p", text)
    def equation(text):
        return ("equation", text)
    def table(headers, rows, widths=None):
        return ("table", (headers, rows, widths))
    def figure(name, caption, height):
        return ("figure", (name, caption, height))

    abstract = (
        "Fixed cache replacement rules can respond differently to streaming, skewed, and changing memory-access patterns. "
        "This study implements a reproducible, trace-driven L1/L2/L3 hierarchy and evaluates a small online logistic reuse predictor at L3. "
        "The predictor samples resident lines and learns only after fixed-horizon reuse labels become observable. "
        f"The evaluation contains {payload['manifest']['run_count']} simulations: six synthetic read-only workloads, three L3 capacities, five seeds, "
        f"and seven policies, with {study['warmup']:,} warmup and {study['measured_requests']:,} measured requests per simulation. "
        f"Across 18 equally weighted workload-capacity cases, online learning changes modeled average memory-access time relative to LRU "
        f"by a mean reduction of {average:.2f}%, ranging from a {best:.2f}% reduction to a {abs(worst):.2f}% increase. "
        f"It improves {wins} cases, ties {ties}, and regresses in {losses}. "
        "A frozen-model ablation exposes sensitivity to workload shifts. Python runtime increases, and predictor latency is not included "
        "in the functional cycle model. The findings support workload-dependent benefits under explicit assumptions, rather than a hardware speedup or a universal improvement."
    )
    section("1 Introduction",
        p("A cache stores a subset of memory blocks near the processor. When the requested block is absent and the target set is full, a replacement policy must choose a resident block to evict. The decision matters because a poor eviction can turn a later request into an access to a slower level. Recency and frequency are useful signals, but their value changes with the access stream."),
        p("This project investigates the broader theme of self-adaptive memory hierarchies through a deliberately narrow control point: L3 victim selection. L1 and L2 retain fixed LRU policies. Demand misses fill upper levels, while the learner decides which L3 resident to remove. The implementation does not control physical processor caches or migrate operating-system pages."),
        p("The research question is whether a lightweight, causally trained reuse predictor can reduce modeled memory-access cost on changing workloads, and what losses and software overhead accompany those decisions. Three hypotheses guide the study: learned reuse helps on skewed or scan-polluted streams; continued learning handles change better than freezing after warmup; and learning introduces measurable runtime and metadata costs."),
        p("The contribution is an inspectable experimental artifact, a bounded delayed-label mechanism, and a paired evaluation including failure cases and ablations. Online reuse prediction and adaptive replacement are established research directions. This manuscript does not assert a first-of-its-kind policy or superiority to the state of the art.")
    )
    section("2 Related Work",
        p("Jaleel et al. introduced re-reference interval prediction, including SRRIP and DRRIP, to improve scan and thrashing behavior [1]. The baseline here implements two-bit SRRIP with insertion at value 2, hit promotion to 0, and aging until an eviction candidate reaches 3. DRRIP is not implemented."),
        p("Hawkeye learns from a reconstruction of Belady's decisions on past cache accesses [2]. It establishes that learned replacement is not a new problem. Our prototype instead uses fixed-horizon binary reuse labels and a software logistic model; it does not reproduce Hawkeye's predictor, sampling scheme, or hardware design."),
        p("LeCaR uses regret minimization to learn between LRU and LFU decisions [3]. Learning Relaxed Belady applies learned replacement in CDN caches [4]. Those studies motivate adaptive policies, but CDN objects and storage workloads differ from fixed-size CPU cache lines. Their published performance numbers are not comparable with this simulator."),
        p("ArtMem applies reinforcement learning to tiered-memory page migration [5]. Its operating-system control and migration granularity differ from the cache-line eviction problem evaluated here. Architectural tools such as ChampSim provide a stronger route to studying execution effects [6]. A future architectural comparison would need modern adaptive baselines and real traces, beyond the conventional baselines in this study.")
    )
    section("3 System Model and Assumptions",
        table(["Component", "Capacity", "Associativity", "Incremental cost"], [
            ["L1", "32 lines / 2 KiB", "4-way", "4 cycles"],
            ["L2", "128 lines / 8 KiB", "8-way", "12 cycles"],
            ["L3", "256, 512, 1024 lines", "8-way", "40 cycles"],
            ["RAM", "All trace addresses", "Not modeled", "200 cycles"],
        ], [0.17, 0.34, 0.2, 0.29]),
        p("The capacities are scaled teaching configurations, and the costs are declared modeling parameters rather than measured latencies. Every line contains 64 bytes. A byte address a maps to block floor(a / 64), and a block maps to set block modulo number_of_sets. Cache ways within that set are the only replacement candidates."),
        p("The hierarchy is read-only and non-inclusive: copies may reside at several levels, and eviction at one level does not invalidate another. Each access probes L1, then L2 on an L1 miss, then L3 on an L2 miss, and RAM on an L3 miss. Every missed cache is filled. Functional insertion can occur before downstream lookup because this read-only model has no timing overlap or invalidation dependencies. Independent, fixed upper caches ensure the same LLC reference stream across L3 policies."),
        equation("AMAT = 4 + 12 m1 + 40 m1 m2 + 200 m1 m2 m3"),
        p("Here m1 is the local L1 miss ratio, m2 is the local L2 miss ratio, and m3 is the local L3 miss ratio. Equivalently, served-by costs are 4, 16, 56, and 256 cycles for L1, L2, L3, and RAM. The implementation accumulates those costs for each measured request. No instruction counts are available, so misses per thousand memory requests must not be described as misses per thousand instructions."),
        equation("Fill bytes = 64 (L1 fills + L2 fills + L3 fills)"),
        p("Fill traffic counts each transfer between adjacent hierarchy levels: an L2-served request transfers 64 bytes, an L3-served request 128 bytes, and a RAM-served request 192 bytes. It excludes tags, replacement metadata, dirty writebacks, prefetches, bandwidth contention, TLBs, coherence, and page faults. There is no proactive migration or operating-system paging.")
    )
    section("4 Online Reuse Policy",
        p("The predictor's clock advances once per request reaching L3. Before each lookup, one resident line from the requested set is sampled uniformly with a seeded generator. Sampling occurs on both hits and misses, so training is not restricted to evicted lines. Empty sets produce no sample. Snapshot features are computed before updating the requested line's access statistics."),
        table(["Feature", "Definition at L3 tick t"], [
            ["Bias", "Constant 1"],
            ["Recency", "min(2, log(1 + age) / log(1 + H))"],
            ["Decayed frequency", "log(1 + f exp(-age / H)) / log(1 + H)"],
            ["Previous interval", "min(2, log(1 + gap) / log(1 + H))"],
            ["Overdue ratio", "min(2, age / max(1, gap)) / 2"],
            ["Resident reuse", "1 if accessed more than once during this residency"],
        ], [0.27, 0.73]),
        p("Age is t minus the last access tick. Gap is the interval preceding the latest access, or zero for a new line. Frequency starts at 1 and is updated on a hit as f = f exp(-gap / H) + 1. Counts and intervals reset on reinsertion; no unbounded per-address history is maintained. The main experiment uses H = 256 L3 requests."),
        equation("p(reuse | x) = 1 / (1 + exp(-w dot x))"),
        p("A snapshot at tick s receives label 1 if its target is referenced in (s, s + H], including the endpoint, and 0 otherwise. Labels are settled only at s + H. Observed requests may mark pending samples positive earlier, but no weight update occurs before the full window closes. Features and the original prediction are saved at sampling time. Prequential Brier and log-loss diagnostics use those saved predictions, and do not evaluate a sample after training on its own outcome."),
        equation("w[j] <- w[j] - 0.05 ((p_current - y) x[j] + lambda[j] w[j])"),
        p("Weights start at zero. A stochastic gradient step uses the current weights when a label matures; lambda is 0 for the bias and 0.0001 for other weights. The learning rate remains constant to allow continued adaptation. For numerical stability logits are clipped to [-30, 30]. At least 64 weight updates are required before learned eviction; LRU supplies the cold-start fallback."),
        p("On an eviction, the policy scores all residents in the requested set and removes the line with minimum reuse probability. Ties use least recency, then block ID. All policies admit every demand miss, so the learned policy is a replacement method rather than a bypass or prefetch controller."),
        ("code", "For each LLC reference at tick t:\n  mark earlier snapshots of the requested block as reused\n  settle snapshots whose endpoint is t; update if enabled\n  sample one current resident and save its features\n  on hit: refresh resident access statistics\n  on full-set miss: evict minimum predicted reuse\n  insert the demanded block; retain pending snapshots"),
        p("At most one snapshot is created per tick and every snapshot expires after H ticks. Thus at most H samples and H distinct target IDs are pending. Evicted blocks can remain pending until their labels mature. At trace end incomplete windows are censored, not converted to negative examples. The predictor uses six weights and O(H) pending metadata, plus O(C3) resident statistics; eviction evaluates O(A3 d) feature terms. This is a logical bound, not a measured hardware SRAM budget, and Python object memory is not reported as hardware storage.")
    )
    section("5 Experimental Method",
        p(f"The declared matrix in configs/paper.json crosses six workloads, L3 capacities of 16, 32, and 64 KiB, seeds {', '.join(map(str, study['seeds']))}, and seven policies. Each simulation processes {study['warmup'] + study['measured_requests']:,} requests. The first {study['warmup']:,} populate caches and train the predictor, and the remaining {study['measured_requests']:,} contribute counters and latency. Cache and model state persist across the warmup boundary. Timing-order effects are reduced by reproducibly shuffling policy order within each workload-size-seed combination."),
        table(["Workload", "Exact generator structure"], [
            ["Sequential", "One pass over distinct cache-line addresses"],
            ["Looping", "Repeated 192-line loop; seed chooses the base offset"],
            ["Uniform random", "Independent uniform references to 4096 lines"],
            ["Zipf", "4096 lines; request weights proportional to rank^-1.15"],
            ["Changing hotspots", "Four equal-length phases; 85% from a disjoint 384-line hotspot, 15% uniform background"],
            ["Hotset + scans", "Each 512-request block: 384 random requests to 192 hot lines, then 128 unique scan lines"],
        ], [0.23, 0.77]),
        p("LRU, FIFO, resident-lifetime LFU, and two-bit SRRIP are conventional baselines. LFU counts reset when a block is evicted and ties use recency. The online variant uses all six features and continues updating. Frozen uses identical features and warmup training, then disables all weight updates during measurement. Recency uses only the bias and recency feature but retains online updates and the same sampling and label mechanism. The frozen variant still gathers and evaluates labels, so its runtime is not an optimized inference-only deployment."),
        p("Every policy receives the identical byte-address trace for its paired seed and workload, verified with SHA-256 hashes. The model receives a current block and cache residents, never the future trace. The horizon and optimizer settings are fixed for the reported matrix; no parameter search or held-out generalization claim is made. Synthetic workloads are fully specified and generated locally. CSV trace import is implemented, but real application traces were not evaluated."),
        p("For each workload-capacity pair, the report averages the five seeds. Relative AMAT changes are computed per paired seed before averaging. A deterministic 2,000-resample percentile bootstrap gives exploratory 95% intervals over those seed pairs. Five seeds provide limited uncertainty information. Sequential and looping cases are effectively deterministic, so zero-width intervals do not imply certainty beyond the generator. The overall mean weights all 18 workload-capacity cases equally; it is not a throughput-weighted production estimate."),
        p("Complete measured-phase wall time is recorded separately from modeled cycles. Internal timers count scalar prediction and weight-update kernels, but exclude some feature extraction, queue management, and bookkeeping; they must not be treated as total learning overhead. Wall times are single executions per case and seed, with shuffled order, rather than controlled repeated microbenchmarks.")
    )
    section("6 Results and Analysis",
        p(f"Across the 18 workload-capacity cases, the online policy achieves a mean paired AMAT reduction of {average:.2f}% relative to LRU. It improves {wins} cases, ties {ties}, and increases latency in {losses}. The range is {worst:+.2f}% to {best:+.2f}% reduction. This arithmetic average includes all cases, including sequential streams, deterministic loops, and negative results. Figure 1 displays the entire matrix rather than selecting only favorable workloads."),
        figure("reduction.png", "Figure 1. Mean paired percentage reduction in modeled AMAT versus LRU across all workloads and L3 capacities. Positive values are improvements; negative values are regressions.", 175),
        table(["Workload", "LRU", "Online", "SRRIP", "Online reduction [95% CI]"], [
            [LABELS[w], f"{lookup(payload,w,512,'lru')['amat_cycles']:.2f}",
             f"{lookup(payload,w,512,'online')['amat_cycles']:.2f}",
             f"{lookup(payload,w,512,'srrip')['amat_cycles']:.2f}",
             f"{lookup(payload,w,512,'online')['amat_reduction_vs_lru_pct']:+.2f}% [{lookup(payload,w,512,'online')['amat_reduction_ci95'][0]:+.2f}, {lookup(payload,w,512,'online')['amat_reduction_ci95'][1]:+.2f}]"]
            for w in study["workloads"]
        ], [0.24, 0.11, 0.11, 0.11, 0.43]),
        p("Table 4 reports mean modeled cycles at 32 KiB L3, with paired seed-bootstrap intervals for the relative reduction. Intervals describe generator-seed variability under this exact setup. They do not account for simulator model error, unseen applications, parameter selection, or hardware costs."),
        figure("absolute_amat.png", "Figure 2. Absolute modeled access cost at 16 and 32 KiB L3, averaged over five seeds. Frozen is the model that stops updating after warmup.", 190),
        p(f"The largest benefit occurs for hotset-plus-scan traffic at 16 KiB: LRU uses {mixed['lru']:.2f} modeled cycles per request, SRRIP {mixed['srrip']:.2f}, and online learning {mixed['online']:.2f}. The stream combines reusable hot blocks with one-time scan pollution. The result is consistent with the predictor retaining useful residents during scans. It does not establish that every scan-heavy application behaves similarly."),
        p("Sequential requests have no reuse, and all policies incur the same 256-cycle served-by-RAM cost. The 192-line loop fits L3 but exceeds L2; after warmup every policy serves it at L3, producing 56 cycles. These controls demonstrate that the learned policy cannot manufacture useful locality. Tiny changes on uniform-random traffic are interpreted as stream-specific variation rather than reliable predictive structure."),
        p(f"At 32 KiB under changing hotspots, online learning uses {phase['online']:.2f} cycles versus LRU's {phase['lru']:.2f}; the learned policy therefore regresses. A plausible explanation is mismatch between the fixed prediction horizon, resident-lifetime features, and phase transition timing. That explanation is a hypothesis rather than an isolated causal finding, because this study does not perform a horizon sensitivity sweep."),
        figure("adaptation.png", "Figure 3. Mean windowed latency for changing hotspots at 32 KiB and hotset-plus-scan traffic at 16 KiB. Dotted vertical lines in the upper panel mark hotspot changes; windows may straddle a boundary.", 230),
        p(f"The frozen policy reaches {phase['frozen']:.2f} cycles on changing hotspots at 32 KiB, versus {phase['online']:.2f} for continued learning and {phase['recency']:.2f} for recency-only learning. Continued updates help relative to this frozen model, yet do not beat LRU. Thus adaptation is useful within the proposed model family, but is insufficient evidence of the best replacement policy."),
        p("The feature ablation changes behavior in a workload-dependent way. Full-feature learning performs better on several skewed or scan-polluted cases, while recency alone stays close to LRU in many cases. The complete seven-policy tables, per-seed outcomes, traffic counters, learning diagnostics, and 1,000-request windows are supplied in the machine-readable results, allowing readers to inspect cases omitted from the compact printed table."),
        figure("runtime.png", "Figure 4. Mean measured Python runtime relative to LRU at 32 KiB L3. This is software execution cost in the prototype, not processor cycle cost or physical speedup.", 145),
        p(f"The online-to-LRU mean-runtime ratio ranges from {min(runtimes):.2f}x to {max(runtimes):.2f}x across the 18 cases. The implementation pays for sampling, delayed labels, features, prediction, training, and Python object bookkeeping. No conversion between wall time and the synthetic cycle model is justified. Consequently a lower modeled AMAT alone cannot establish a net execution-time benefit after inference and training costs.")
    )
    section("7 Validation and Reproducibility",
        p("Automated tests cover hand-calculated hierarchy outcomes, line offsets and set conflicts, exact victim decisions, warmup exclusion with retained state, endpoint-inclusive delayed labels, censoring, frozen-model behavior, metadata bounds, trace validation, and counter conservation. They also verify that paired policies share trace hashes and identical upper-cache miss counts. These are functional checks of the implementation, not validation of the model against a real processor."),
        p("The artifact includes the source, fixed experiment configuration, CSV imports, a local dashboard, raw.csv, summary.csv, benchmark.json, and a manifest containing Python/platform details and SHA-256 hashes of simulator, configuration, and test files. Scientific figures and this manuscript are generated from benchmark.json. A source-stability flag records whether the hashed code changed while the experiment was running."),
        ("code", "python -m pytest -q\npython -m adaptive_memory benchmark --config configs/paper.json\npython -m adaptive_memory report\npython -m adaptive_memory serve --port 8765"),
        p("The simulator and dashboard use Python's standard library. Figure and PDF generation require the packages in requirements.txt. Reproductions should use a clean Python environment and record their own timings; numerical cache outcomes are deterministic for a fixed Python implementation, trace, seed, and source version. Floating-point learning implementations on other runtimes may differ slightly.")
    )
    section("8 Limitations and Future Work",
        p("External validity is limited by short synthetic read-only traces, scaled caches, a serial latency model, and five seeds. The model excludes instruction execution, memory-level parallelism, realistic bandwidth, writebacks, coherence, address translation, prefetching, and hardware predictor delays. It cannot support claims about IPC, energy, real page-fault reduction, or operating-system data migration."),
        p("The baselines are useful educational references but do not establish state-of-the-art competitiveness. Neither Hawkeye, DRRIP, nor a faithfully reproduced LeCaR variant is evaluated. The learner has no admission control, confidence-based fallback after cold start, explicit phase detector, or net-cost objective. Hyperparameter sensitivity, training-window sensitivity, and sample-rate sensitivity remain untested."),
        p("A stronger follow-up would evaluate licensed or publicly distributable application traces, add stronger replacement baselines, sweep horizon and sampling budgets on separate development workloads, measure metadata in a concrete implementation, and evaluate the chosen policy in ChampSim or another architectural simulator. Hardware feasibility would require a specified predictor implementation and accounting for inference latency, energy, and storage. A separate DRAM/CXL or DRAM/SSD page-placement project would need a page migration model and migration-cost experiments.")
    )
    section("9 Conclusion",
        p(f"A causal online reuse predictor can improve replacement in selected synthetic cache workloads, with the largest observed modeled AMAT reduction of {best:.2f}% relative to LRU. The complete matrix yields only {average:.2f}% mean reduction and includes {losses} regressing cases. Continued learning outperforms its frozen counterpart on the reported changing-hotspot example, while conventional policies remain stronger in some configurations. The resulting project is a reproducible functional research prototype. The evidence motivates further evaluation and cost modeling; it does not establish a universally better or hardware-ready memory architecture.")
    )
    return abstract, sections


def write_markdown(abstract, sections, path):
    parts = ["# Online Learning for Adaptive Cache Replacement Under Changing Access Patterns", "",
             "Research manuscript | November 21, 2026", "", "## Abstract", "", abstract, "",
             "Keywords: cache replacement; memory hierarchy; online learning; reuse prediction; trace-driven simulation.", ""]
    table_number = 0
    captions = {1: "Scaled cache hierarchy and declared serial lookup costs.",
                2: "Predictor features, with H denoting the reuse horizon.",
                3: "Synthetic workload definitions.",
                4: "Mean modeled cycles and paired reductions at 32 KiB L3."}
    for heading, blocks in sections:
        parts.extend([f"## {heading}", ""])
        for kind, data in blocks:
            if kind == "table":
                table_number += 1
                parts.extend([f"Table {table_number}. {captions[table_number]}", ""])
                headers, rows, _ = data
                parts.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
                parts.extend("| " + " | ".join(row) + " |" for row in rows)
            elif kind == "figure":
                name, caption, _ = data
                parts.extend([f"![{caption}](figures/{name})", "", caption])
            elif kind in ("equation", "code"):
                parts.extend(["```text", data, "```"])
            else:
                parts.append(data)
            parts.append("")
    parts.extend(["## References", ""])
    parts.extend(f"[{i}] {text} [Source]({url})\n" for i, (_, text, url) in enumerate(REFERENCES, 1))
    path.write_text("\n".join(parts), encoding="utf-8")


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in value)


def write_latex(abstract, sections, path):
    content = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=22mm]{geometry}",
               r"\usepackage[T1]{fontenc}", r"\usepackage{mathptmx,graphicx,booktabs,tabularx,hyperref}",
               r"\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}",
               r"\title{Online Learning for Adaptive Cache Replacement Under Changing Access Patterns}",
               r"\author{}", r"\date{November 21, 2026}", r"\begin{document}", r"\maketitle",
               r"\begin{abstract}", tex_escape(abstract), r"\end{abstract}",
               r"\noindent\textbf{Keywords:} cache replacement, memory hierarchy, online learning, reuse prediction."]
    table_number = 0
    captions = {1: "Scaled cache hierarchy and declared serial lookup costs.",
                2: "Predictor features, with H denoting the reuse horizon.",
                3: "Synthetic workload definitions.",
                4: "Mean modeled cycles and paired reductions at 32 KiB L3."}
    for heading, blocks in sections:
        content.append(r"\section*{" + tex_escape(heading) + "}")
        for kind, data in blocks:
            if kind == "table":
                table_number += 1
                headers, rows, _ = data
                content += [r"\begin{table}[htbp]\centering\small", r"\caption{" + tex_escape(captions[table_number]) + "}", r"\begin{tabularx}{\linewidth}{" + "X" * len(headers) + "}", r"\toprule",
                            " & ".join(tex_escape(h) for h in headers) + r" \\", r"\midrule"]
                content.extend(" & ".join(tex_escape(cell) for cell in row) + r" \\" for row in rows)
                content += [r"\bottomrule", r"\end{tabularx}\end{table}"]
            elif kind == "figure":
                name, caption, _ = data
                # PDF/Markdown carry numbered captions; LaTeX numbers the same figures itself.
                caption = caption.split(". ", 1)[1]
                content += [r"\begin{figure}[htbp]\centering", r"\includegraphics[width=\linewidth]{figures/" + name + "}",
                            r"\caption{" + tex_escape(caption) + "}", r"\end{figure}"]
            elif kind in ("code", "equation"):
                content += [r"\begin{verbatim}", data, r"\end{verbatim}"]
            else:
                content += [tex_escape(data), ""]
    content += [r"\begin{thebibliography}{9}"]
    for key, text, url in REFERENCES:
        content += [r"\bibitem{" + key + "} " + tex_escape(text) + r" \url{" + url + "}"]
    content += [r"\end{thebibliography}", r"\end{document}"]
    path.write_text("\n".join(content), encoding="utf-8")


def write_pdf(abstract, sections, output, figures_dir):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Preformatted, KeepTogether

    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    usable = width - 100
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", fontName="Times-Bold", fontSize=19, leading=23, alignment=TA_CENTER, textColor=colors.black, spaceAfter=11))
    styles.add(ParagraphStyle(name="PaperMeta", fontName="Helvetica", fontSize=8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#4e626c"), spaceAfter=15))
    styles.add(ParagraphStyle(name="PaperBody", fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=7))
    styles.add(ParagraphStyle(name="PaperAbstract", fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=9))
    styles.add(ParagraphStyle(name="PaperH1", fontName="Times-Bold", fontSize=12.5, leading=16, textColor=colors.black, spaceBefore=13, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name="PaperCell", fontName="Helvetica", fontSize=8, leading=11, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="PaperHeader", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white))
    styles.add(ParagraphStyle(name="PaperCaption", fontName="Times-Italic", fontSize=9, leading=12, spaceAfter=11))
    styles.add(ParagraphStyle(name="PaperCode", fontName="Courier", fontSize=8, leading=11, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(name="PaperRef", fontName="Times-Roman", fontSize=9, leading=12, spaceAfter=7, wordWrap="CJK"))
    story = [Paragraph("Online Learning for Adaptive Cache Replacement<br/>Under Changing Access Patterns", styles["PaperTitle"]),
             Paragraph("Research manuscript · November 21, 2026", styles["PaperMeta"]),
             Paragraph("<b>Abstract</b> — " + escape(abstract), styles["PaperAbstract"]),
             Paragraph("<b>Keywords:</b> cache replacement; memory hierarchy; online learning; reuse prediction; trace-driven simulation.", styles["PaperAbstract"])]
    table_number = 0
    for heading, blocks in sections:
        story.append(Paragraph(escape(heading), styles["PaperH1"]))
        for kind, data in blocks:
            if kind == "table":
                table_number += 1
                headers, rows, proportions = data
                values = [[Paragraph(escape(cell), styles["PaperHeader"]) for cell in headers]]
                values.extend([[Paragraph(escape(cell), styles["PaperCell"]) for cell in row] for row in rows])
                widths = [usable * n for n in proportions] if proportions else None
                item = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
                item.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243d49")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f4f4")]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9d9d9")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                captions = {1: "Table 1. Scaled cache hierarchy and declared serial lookup costs.",
                            2: "Table 2. Predictor features, with H denoting the reuse horizon.",
                            3: "Table 3. Synthetic workload definitions.",
                            4: "Table 4. Mean modeled cycles and paired reductions at 32 KiB L3."}
                story.append(KeepTogether([Paragraph(captions.get(table_number, f"Table {table_number}."), styles["PaperCaption"]), item, Spacer(1, 10)]))
            elif kind == "figure":
                name, caption, requested_height = data
                image = Image(str(figures_dir / name))
                aspect = image.imageHeight / image.imageWidth
                image.drawWidth = usable
                image.drawHeight = usable * aspect
                story.append(KeepTogether([image, Paragraph(escape(caption), styles["PaperCaption"])]))
            elif kind in ("code", "equation"):
                story.append(Preformatted(data, styles["PaperCode"]))
            else:
                story.append(Paragraph(escape(data), styles["PaperBody"]))
    reference_blocks = [Paragraph("References", styles["PaperH1"])]
    for i, (_, text, url) in enumerate(REFERENCES, 1):
        linked = f'[{i}] {escape(text)} <link href="{escape(url)}" color="#176b53">Source</link>'
        reference_blocks.append(Paragraph(linked, styles["PaperRef"]))
    story.append(KeepTogether(reference_blocks))

    def page_footer(canvas, doc):
        canvas.setTitle("Online Learning for Adaptive Cache Replacement Under Changing Access Patterns")
        canvas.setAuthor("")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#647780"))
        canvas.drawString(50, 30, "Adaptive Memory Lab | Functional simulation study")
        canvas.drawRightString(width - 50, 30, str(doc.page))

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=50, leftMargin=50,
                            topMargin=43, bottomMargin=48, title="Adaptive Cache Replacement", author="")
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def build_report(results_path):
    path = Path(results_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    study = payload["manifest"]["study"]
    if set(study["workloads"]) != set(LABELS) or set(study["l3_lines"]) != {256, 512, 1024}:
        raise ValueError("Paper generator requires the complete configs/paper.json workload/capacity matrix")
    if set(study["policies"]) != set(POLICY_LABELS):
        raise ValueError("Paper generator requires all seven policy variants")
    from .config import Config
    values = dict(study.get("simulator", {}))
    values["l3_lines"] = 512
    if Config(**values) != Config() or len(study["seeds"]) != 5:
        raise ValueError("Paper prose requires the default simulator parameters and exactly five seeds")
    paper_dir = ROOT / "paper"
    paper_dir.mkdir(exist_ok=True)
    figure_dir = paper_dir / "figures"
    figures(payload, figure_dir)
    abstract, sections = manuscript(payload)
    write_markdown(abstract, sections, paper_dir / "research_paper.md")
    write_latex(abstract, sections, paper_dir / "research_paper.tex")
    write_pdf(abstract, sections, ROOT / "output/pdf/research_paper.pdf", figure_dir)
    bib = "\n\n".join("@misc{" + key + ",\n  note = {" + text.replace("'", "") + "},\n  url = {" + url + "}\n}" for key, text, url in REFERENCES)
    (paper_dir / "references.bib").write_text(bib + "\n", encoding="utf-8")
    provenance = {"results_file": str(path.resolve()), "results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "run_count": len(payload["runs"]), "paper_title": "Online Learning for Adaptive Cache Replacement Under Changing Access Patterns",
                  "generated_files": ["paper/research_paper.md", "paper/research_paper.tex", "output/pdf/research_paper.pdf"]}
    (paper_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Created manuscript, LaTeX source, figures, and PDF from {len(payload['runs'])} measured runs.")
