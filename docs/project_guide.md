# Understanding and Demonstrating Adaptive Memory Lab

## The problem in simple language

Imagine a study desk with limited space and a large bookshelf. Frequently used books are useful on the desk because fetching them from the shelf takes longer. When the desk is full, you must decide which book to remove. A cache replacement policy makes this kind of decision for fixed-size blocks of memory.

The project studies three simulated cache levels. L1 is the smallest and cheapest to access; L2 and L3 are progressively larger and more expensive. RAM holds all requested addresses in this model. The learning component controls eviction at L3. It never changes the contents of your real processor's cache.

## Follow one request

Suppose the CPU requests byte address 128. With 64-byte lines, it belongs to block 2. If L1 has 8 sets, block 2 maps to set 2. A cache looks for the block only among the ways of that set. This is called set associativity.

If L1 misses, the request continues to L2. If L2 also misses but L3 hits, the model charges 4 + 12 + 40 = 56 cycles. The block is copied into L1 and L2, contributing 128 fill bytes. If every cache misses, RAM serves the request, three caches are filled, and the model charges 256 cycles and 192 fill bytes.

Copies may exist at more than one level. The hierarchy is non-inclusive: evicting from L3 does not remove an L1 or L2 copy. This model keeps the LLC request stream independent of L3's policy.

## Why use machine learning

LRU remembers when each resident was last used. LFU counts how often each resident has been used during its current stay. These are useful rules, but a stream containing popular blocks and one-time scan blocks may require a different balance.

The model learns a score from several features: recency, decayed frequency, the last inter-access gap, whether the block is overdue relative to that gap, and whether it has already been reused during this cache residency. The probability estimates reuse within a fixed window of future LLC references.

The model is logistic regression, not a large language model and not deep learning. It uses six coefficients and a sigmoid function. Training adjusts coefficients after observing outcomes. Small models make this logic easier to inspect, but the Python prototype is still slower to execute than simple policies.

## Why delayed labels matter

At tick 100, the model might snapshot block A and ask whether A will be referenced in the next 256 LLC requests. The label is only final at tick 356. If A appears at tick 200, the pending example is marked positive, but training still waits for tick 356. If A never appears in that window, the completed label is negative.

A simulator has the whole trace in memory. Giving that future sequence to the online policy would give it information a real system lacks. The policy interface therefore receives only the current block, current residents, and current tick. The last incomplete windows are discarded as censored observations.

## How to demonstrate the project

1. Start the dashboard with `start_dashboard.bat`.
2. Load the saved paper results and choose **Hotset + streaming scans**, **16 KiB** L3. Explain that hot data and one-time scan data compete for a small cache.
3. Compare LRU, SRRIP, and Online in the chart and table. Values are simulated costs; Python runtime is a separate metric.
4. Choose **Changing hotspots**, **32 KiB** L3. Show the learner's regression against LRU. Explain that adaptive does not mean universally better.
5. Point to Frozen in the table. Freezing training is an ablation that tests whether ongoing adaptation matters for this particular model family.
6. Change the seed or horizon and click **Run experiment**. Explain that this is a new single-seed demo, not the paper's five-seed average.
7. Restore paper results and open the paper. Show the experiment configuration, raw data, and limits.

## Questions you should be able to answer

**What exactly is your contribution?**

A working, causally trained cache-replacement prototype and a reproducible evaluation with ablations and negative results. The broad idea of learned cache replacement is established. A publishable novelty claim would require a more extensive literature review and stronger comparative evaluation.

**Does the model predict the next exact address?**

No. It predicts a binary event: whether an existing block will be reused within H LLC references. It scores residents of the set that needs an eviction.

**Is the model pretrained?**

No. It starts with zero weights. Warmup supplies initial training examples. The full model keeps learning after warmup, while the frozen variant stops updating. LRU supplies victim decisions until at least 64 weight updates have occurred.

**Why not reinforcement learning?**

A supervised reuse target is easy to define and audit. Reinforcement learning would require choices about state, actions, reward delays, and exploration, making it harder to separate simulator correctness from learning behavior. This project does not claim logistic regression is the best possible learner.

**Does a higher prediction accuracy guarantee fewer misses?**

No. Class balance, the horizon, and the costs of individual errors matter. A good average classifier can rank eviction candidates poorly. The main outcomes are actual simulated cache behavior; Brier score and log loss are diagnostic measurements.

**Does a lower modeled AMAT mean the real computer runs faster?**

No. The model omits hardware inference cost, instruction execution, memory-level parallelism, and contention. Python runtime measures the prototype's software cost. A hardware speedup claim needs architectural evaluation and an implementable predictor cost model.

**Why are page faults absent?**

Cache misses and page faults describe different layers. This simulator maps cache lines and has a RAM backing store; it does not model virtual-memory pages, page tables, or disk-backed swapping.

**What does the bounded memory claim mean?**

There is at most one new sample per LLC reference, and each expires after H references. Therefore no more than H samples and H distinct pending target IDs remain. The weights are constant sized, and resident statistics scale with cache capacity. This is a logical object-count bound, not a measured hardware memory budget.

**What is missing for a stronger research paper?**

Real application traces; stronger baselines such as DRRIP or a faithful Hawkeye implementation; independent development/test workloads; longer runs; parameter sensitivity; concrete metadata and inference cost; and architectural validation. These remain future work, rather than completed experiments.

