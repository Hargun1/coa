# Adaptive Memory Lab

A working COA research project on **self-adaptive memory hierarchies using online learning**. The project simulates L1, L2, L3, and RAM and learns which cache line to evict from L3. The accompanying paper is generated from actual local simulation results.

**Scope:** a read-only, trace-driven functional simulator. It does not modify your CPU's caches, control operating-system memory, or demonstrate hardware speedup. All supplied workloads are synthetic. The learner sometimes loses to conventional policies.

## Open the project

On Windows, double-click **`start_dashboard.bat`**, then open **http://127.0.0.1:8765**. Python 3.11+ is required. The simulator and dashboard use only the standard library; Docker, a GPU, and paid APIs are unnecessary. Stop the server with Ctrl+C.

Alternatively, from this folder:

```powershell
python -m adaptive_memory serve --port 8765
```

The dashboard can display the saved paper results or run seven policies on a fresh synthetic trace. Saved paper results average five seeds; an interactive experiment uses one seed. Select the access pattern and L3 capacity, choose a reuse horizon and seed, and click **Run experiment**. **Restore paper results** returns to the measured study. Changing the horizon affects new experiments, not saved results.

## Research paper

- [Paper PDF](output/pdf/research_paper.pdf)
- [Editable manuscript](paper/research_paper.md)
- [LaTeX source](paper/research_paper.tex), suitable for adaptation to your faculty's required template
- [Literature and verified source links](docs/literature.md)
- [Project explanation and viva preparation](docs/project_guide.md)

The PDF is a research manuscript, not an accepted publication or an exact IEEE conference template. Add the author names, affiliation, faculty details, and any required declaration before submission. Novelty against the full literature has not been established. The LaTeX source is supplied for editing; the delivered PDF is built with ReportLab, and the LaTeX has not been compiled here.

## What is implemented

```text
Byte-address trace
     |
     v
L1 [LRU, 4 ways] --miss--> L2 [LRU, 8 ways]
                               |
                              miss
                               v
                       L3 [policy under test, 8 ways]
                               |
                              miss
                               v
                              RAM
```

Each missed cache receives a demand fill. The caches are non-inclusive, so an eviction does not invalidate another cache's copy. Upper caches are identical across policies, preserving the LLC input stream for fair comparisons.

| Policy | Eviction rule |
| --- | --- |
| LRU | Least recently accessed resident |
| FIFO | Earliest inserted resident |
| LFU | Lowest resident-lifetime access count, with LRU tie break |
| SRRIP | Two-bit static re-reference interval prediction |
| Online | Six-feature logistic reuse prediction, updated throughout the run |
| Frozen | Same model; updates stop after warmup |
| Recency | Online logistic model using only bias and recency |

The learner samples one resident of the requested L3 set per LLC access. A label says whether that block is reused in the next H LLC requests. **It learns only when that entire window has passed.** Samples at the end without a complete window are censored. At most H pending samples and H target IDs are retained. All weights are initialized to zero; cold-start victim decisions use LRU until 64 labels have trained the model.

## Run one simulation

```powershell
python -m adaptive_memory simulate --workload mixed_scan --policy online --l3-lines 256 --requests 15000 --warmup 3000
python -m adaptive_memory simulate --workload phase_shift --policy lru --output results/single.json
python -m adaptive_memory simulate --trace examples/tiny_trace.csv --policy lru --warmup 0
```

CSV input accepts an `address` column and an optional `operation` column. Only `R` and `READ` are supported. Decimal byte addresses and `0x` hexadecimal byte addresses are accepted; negative or out-of-range addresses and write operations are rejected. Headerless one-address-per-line files also work. Importing a real trace is supported, but no real application trace is included in the reported study.

## Reproduce the study and paper

For the first setup, create a virtual environment and install the tested top-level package versions:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m adaptive_memory benchmark --config configs/paper.json --output results
.venv\Scripts\python.exe -m adaptive_memory report
```

The matrix has **630 simulations**: 6 workloads × 3 L3 capacities × 5 seeds × 7 policies. Each contains 3,000 warmup and 12,000 measured requests. Warmup fills caches and trains models but does not contribute measured hit or latency counters. State persists after warmup. Experiment runtime depends on your machine.

The paper generator expects this complete matrix, five seeds, and the declared default model parameters. It deliberately rejects incomplete or incompatible studies so the prose cannot silently misdescribe another experiment. If you change the scientific design, update the manuscript generator as well.

| File | Contents |
| --- | --- |
| `results/benchmark.json` | Complete results, request windows, learning diagnostics, and manifest |
| `results/raw.csv` | One row per simulation |
| `results/summary.csv` | Per-workload/capacity/policy means and paired seed-bootstrap intervals |
| `results/manifest.json` | Study configuration, platform, Python version, source hashes |
| `paper/provenance.json` | Hash of the input results used to generate the manuscript |
| `paper/figures/` | Standalone scientific figures |
| `output/verification.json` | Checks performed for the supplied artifact |

To verify stored results and their source hashes:

```powershell
python scripts/verify_artifacts.py
```

To compile the editable LaTeX on your own machine, run `pdflatex research_paper.tex` twice from `paper/`, or upload `paper/` to Overleaf. It uses a generic article layout, which should be adapted to your submission venue.

## Interpret the results correctly

The baseline serial lookup costs are incremental: L1 4, L2 +12, L3 +40, RAM +200 cycles. A request served by RAM costs **256 modeled cycles**, not 200. These values are declared teaching assumptions.

```text
AMAT = 4 + 12*m1 + 40*m1*m2 + 200*m1*m2*m3
```

Each `m` is the local miss ratio at that level. The implementation accumulates the equivalent served-by cost directly.

- **Modeled AMAT** describes cache behavior under the fixed cost model. It excludes predictor execution cost.
- **Python runtime** measures actual software execution on the host. It includes the simulation's Python overhead and is reported separately.
- **Cache hit ratio** counts requests served by any cache. The per-level local ratios are also stored.
- **Fill bytes** count transfers across adjacent cache levels. RAM-served requests fill three caches and count 192 bytes.
- **Intervals** are exploratory paired-seed bootstrap intervals. Five seeds do not establish generality across real applications.

For the supplied matrix, the learner has both improvements and regressions. Read the full results and limitations in the paper. Do not describe simulated latency reductions as measured CPU speedups, reduced energy, fewer page faults, or reduced operating-system migration.

## Files to study first

1. `adaptive_memory/simulator.py` — address mapping, cache behavior, hierarchy accounting.
2. `adaptive_memory/policies.py` — conventional policies and delayed online learning.
3. `adaptive_memory/workloads.py` — reproducible access streams and CSV import.
4. `adaptive_memory/experiments.py` — paired experiments and statistical summaries.
5. `adaptive_memory/reporting.py` — figures and data-derived manuscript generation.

For a publication-strength extension, add real application traces, modern adaptive baselines, sensitivity studies, and architectural evaluation with a concrete predictor overhead model.

