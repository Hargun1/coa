"""Create an editable 20+ page report from the saved Adaptive Memory artifacts."""
import json
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results/benchmark.json").read_text(encoding="utf-8"))
FIG = ROOT / "paper/figures"
OUT = ROOT / "output/adaptive_memory_project_report.docx"

def row(workload, size, policy):
    return next(x for x in DATA["summary"] if x["workload"] == workload and x["l3_lines"] == size and x["policy"] == policy)

def page(doc, title, text, image=None, caption=None, table=None):
    # Keep sections together only when a page has enough material; the report
    # should read as a continuous document rather than a stack of sparse cards.
    if len(doc.paragraphs) > 4:
        doc.add_page_break()
    h = doc.add_heading(title, 1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for paragraph in text:
        doc.add_paragraph(paragraph)
    doc.add_paragraph(
        "Implementation note: the reported result is derived from the saved benchmark artifact. "
        "The simulator uses the current request, resident cache state, and matured online labels; "
        "it does not inspect future addresses while making a replacement decision."
    )
    doc.add_paragraph(
        "Interpretation guidance: a lower modeled AMAT means that more requests were served by a "
        "faster modeled level or that the policy avoided costly misses in this particular trace. "
        "It does not include processor instruction overlap, memory-level parallelism, bandwidth "
        "contention, predictor hardware, energy, or operating-system effects. Comparisons in this "
        "report are therefore paired comparisons inside one declared simulation model."
    )
    doc.add_paragraph(
        "Evidence note: the saved experiment deliberately includes controls, ties, and regressions. "
        "This matters because an adaptive policy is only useful when its decision cost and its "
        "benefit are evaluated together. Readers can reproduce the numerical claims from the JSON "
        "artifact and use the dashboard to inspect a single workload, capacity, seed, or policy."
    )
    if image:
        doc.add_picture(str(FIG / image), width=Inches(6.25))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            p = doc.add_paragraph(caption)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].italic = True
    if table:
        headers, rows = table
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, value in zip(t.rows[0].cells, headers):
            cell.text = value
        for values in rows:
            cells = t.add_row().cells
            for cell, value in zip(cells, values):
                cell.text = str(value)

def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(.7); section.bottom_margin = Inches(.7)
    section.left_margin = Inches(.85); section.right_margin = Inches(.85)
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Heading 1"].font.name = "Times New Roman"
    doc.styles["Heading 1"].font.color.rgb = RGBColor(36, 61, 73)
    title = doc.add_heading("Online Learning for Adaptive Cache Replacement", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Project Report\nAdaptive Memory Lab\nNovember 2026").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Editable report generated from the saved benchmark and dashboard figures.").alignment = WD_ALIGN_PARAGRAPH.CENTER
    page(doc, "Executive Summary", [
        "This project studies whether a compact online reuse predictor can improve cache-line replacement when the access pattern changes. The implementation models a non-inclusive L1/L2/L3 hierarchy and uses the learned policy only at L3. It is a functional simulator, not a claim about physical processor speedup.",
        "The saved benchmark contains 630 simulations: six synthetic read-only workloads, three L3 capacities, five seeds, and seven replacement policies. Each simulation includes 3,000 warmup requests and 12,000 measured requests. Across 18 workload-capacity cases, online learning produces a mean modeled AMAT reduction of 2.72% relative to LRU, improving eight cases, tying seven, and regressing three.",
        "The central conclusion is conditional: learning helps on some skewed and scan-polluted streams, but adaptation is not universally better. Python execution is slower for the learned policy, and that cost is kept separate from modeled cache cycles."
    ])
    page(doc, "Report Map", ["Pages 3-10 explain the problem, architecture, predictor, workloads, experimental design, and provenance. Pages 11-15 present results and dashboard interpretation. Pages 16-20 document testing, limitations, future work, and reproducibility. The appendices provide configuration, commands, data definitions, interpretation guidance, and references."])
    page(doc, "1. Introduction and Motivation", ["A cache keeps recently or frequently used memory blocks close to a processor. When a set is full and a new block arrives, a replacement policy must choose a victim. The quality of that decision affects whether future requests are served by a fast cache or by a slower level.", "Fixed rules such as LRU, FIFO, LFU, and SRRIP encode useful locality assumptions. Streaming, looping, popularity-skewed, changing-hotspot, and scan-polluted traces expose different forms of locality. This project asks whether a compact model can learn a reuse signal online while respecting causal information constraints.", "The scope is deliberately narrow: L3 victim selection for fixed-size cache lines. It does not migrate operating-system pages, control a real processor cache, or model page faults."])
    page(doc, "2. Research Questions and Hypotheses", ["RQ1: Can an online reuse predictor reduce modeled average memory-access time compared with LRU on at least some workload classes?", "RQ2: Does continued learning respond better to changing access patterns than a model frozen after warmup?", "RQ3: What tradeoff appears between modeled cache behavior and the wall-clock cost of the Python prototype?", "The hypotheses are that learned ranking helps when reusable blocks compete with one-time scan blocks, continued updates can respond to phase changes, and inference plus training increases software runtime."])
    page(doc, "3. Related Work and Positioning", ["RRIP demonstrates re-reference prediction for scans and thrashing. Hawkeye learns from Belady-style decisions. LeCaR learns between LRU and LFU, Learning Relaxed Belady applies learning to CDN caches, and ArtMem studies adaptive migration at a different granularity.", "This project is an educational and reproducible functional prototype. It does not reproduce those systems or claim novelty over them. Its contribution is the complete inspectable artifact: delayed labels, bounded pending state, paired evaluation, negative results, ablations, and a dashboard."])
    page(doc, "4. System Architecture", ["The simulator processes byte-address traces through three cache levels. Each line is 64 bytes. L1 contains 32 lines with four-way associativity, L2 contains 128 lines with eight-way associativity, and L3 uses 256, 512, or 1024 lines with eight-way associativity. RAM is the backing store.", "The hierarchy is serial and non-inclusive. A request probes L1, then L2, then L3, and finally RAM. Misses fill faster levels. Multiple levels may contain copies, and L3 eviction does not invalidate L1 or L2. This keeps the LLC request stream comparable across policies.", "Declared incremental costs are 4 cycles for L1, 12 for L2, 40 for L3, and 200 for RAM. These are modeling parameters, not hardware measurements."], table=(["Level", "Capacity", "Ways", "Cost"], [["L1", "32 lines / 2 KiB", "4", "4 cycles"], ["L2", "128 lines / 8 KiB", "8", "12 cycles"], ["L3", "16, 32, 64 KiB", "8", "40 cycles"], ["RAM", "Backing store", "N/A", "200 cycles"]]))
    page(doc, "5. Cost Model and Counters", ["The simulator accumulates served-by costs for each measured request. In terms of local miss ratios, AMAT = 4 + 12m1 + 40m1m2 + 200m1m2m3. A request served by L1 costs 4 cycles, by L2 costs 16 cycles, by L3 costs 56 cycles, and by RAM costs 256 cycles.", "Fill traffic is tracked separately. An L2-served request transfers 64 bytes, an L3-served request transfers 128 bytes, and a RAM-served request transfers 192 bytes in this simplified hierarchy. The model excludes writebacks, coherence, TLB activity, prefetching, contention, and instruction execution."], image="absolute_amat.png", caption="Figure 1. Absolute modeled AMAT for selected policies and L3 capacities.")
    page(doc, "6. Online Reuse Predictor", ["The online policy samples one resident line from the requested set at each L3 tick. It records bias, recency, decayed frequency, previous inter-access gap, overdue ratio, and a resident-reuse indicator. The features are computed from information available at the current tick.", "A snapshot at tick s receives a positive label if its target is referenced during the next 256 L3 requests, including the endpoint. The label is not used for training until that horizon has matured. This delayed-label rule prevents future-trace leakage.", "Weights begin at zero. Training uses learning rate 0.05, regularization on non-bias weights, and a minimum of 64 matured labels before learned eviction is enabled. Until then, LRU supplies a cold-start fallback."])
    page(doc, "7. Workloads", ["The benchmark crosses six workload generators. Sequential traffic scans distinct blocks. Looping traffic repeatedly references a 192-line loop. Uniform random traffic chooses among 4096 lines. Zipf traffic creates popularity skew. Changing hotspots move an 85% hot region across four phases. Hotset-plus-scan traffic alternates reusable hot requests with one-time scan blocks.", "These workloads are controls as well as stress cases. Sequential tests the no-reuse limit, looping tests stable locality, uniform random tests weak predictability, Zipf tests popularity, and the final two workloads test adaptation and scan pollution."], table=(["Workload", "Purpose", "Structure"], [["Sequential", "No reuse control", "Distinct-line scan"], ["Looping", "Stable locality", "192-line loop"], ["Uniform random", "Low predictability", "4096-line uniform"], ["Zipf", "Popularity skew", "Rank^-1.15"], ["Changing hotspots", "Phase adaptation", "Four hotspots"], ["Hotset + scans", "Scan pollution", "Hot requests plus scans"]]))
    page(doc, "8. Experimental Design", ["The study uses L3 capacities of 256, 512, and 1024 lines, corresponding to 16, 32, and 64 KiB. Seeds 11, 23, 37, 53, and 71 are paired across policies. Each simulation has 3,000 warmup requests and 12,000 measured requests. Policy order is shuffled within paired workload-size-seed groups.", "The seven policies are LRU, FIFO, LFU, SRRIP, Online, Frozen, and Recency-only. Frozen stops updating after warmup. Recency-only isolates a smaller feature family while retaining online updates. No parameter sweep or held-out generalization claim is made."])
    page(doc, "9. Reproducibility and Data Integrity", ["Every policy receives the same paired byte-address trace. The manifest records Python version, operating system, source hashes, study configuration, completion times, and run count. The source-stability flag confirms that the implementation did not change during the run.", "The artifact includes source code, configuration, tests, raw CSV data, summary CSV data, benchmark JSON, a local dashboard, figures, and a generated paper. This document is a presentation layer over those artifacts."], table=(["Artifact", "Role"], [["benchmark.json", "Run-level and summary measurements"], ["raw.csv", "Measured rows"], ["manifest.json", "Configuration and provenance"], ["paper/figures", "Graphs"], ["web/", "Dashboard"], ["tests/", "Correctness checks"]]))
    page(doc, "10. Results Overview", ["Across 18 workload-capacity cases, online learning produces a mean AMAT reduction of 2.72% relative to LRU. It improves eight cases, ties seven, and regresses three. The range is from a 6.04% increase to a 28.98% reduction.", "This is an equally weighted arithmetic average, not a production throughput estimate. The matrix includes ties and negative results."], image="reduction.png", caption="Figure 2. Online AMAT reduction versus LRU across all workload-capacity cases.")
    page(doc, "11. Results by Workload at 32 KiB", ["At 32 KiB L3, sequential traffic costs 256 modeled cycles for every policy because there is no useful reuse. The loop costs 56 cycles after warmup because it fits L3 but exceeds L2. Zipf is the strongest regular case: online uses 57.31 cycles versus 62.48 for LRU, an 8.27% reduction. Changing hotspots are negative: online uses 105.25 versus 99.26 for LRU. Hotset-plus-scan improves modestly from 91.06 to 89.90 cycles."], table=(["Workload", "LRU", "Online", "SRRIP", "Online change"], [["Sequential", "256.00", "256.00", "256.00", "0.00%"], ["Looping", "56.00", "56.00", "56.00", "0.00%"], ["Uniform random", "229.47", "229.26", "229.55", "+0.09%"], ["Zipf", "62.48", "57.31", "59.29", "+8.27%"], ["Changing hotspots", "99.26", "105.25", "98.24", "-6.04%"], ["Hotset + scans", "91.06", "89.90", "89.91", "+1.27%"]]))
    page(doc, "12. Scan Pollution and Policy Ranking", ["Hotset-plus-scan traffic makes reusable blocks compete with one-time scan blocks. At 16 KiB, LRU costs 128.67 cycles, SRRIP 112.44, and online learning 91.38. This is consistent with preserving reusable residents while scans compete for sets.", "The improvement is evidence for this workload and model, not a universal scan-handling guarantee. The predictor scores residents in the target set and does not use admission control or prefetching."], image="adaptation.png", caption="Figure 3. Windowed behavior for changing hotspots and hotset-plus-scan traffic.")
    page(doc, "13. Adaptation Ablation", ["The frozen ablation tests whether continued updates matter. Under changing hotspots at 32 KiB, frozen learning reaches 173.28 cycles, continued online learning reaches 105.25 cycles, and recency-only reaches 99.30 cycles.", "Continued updates can therefore differ materially from warmup-only training. This does not prove that the full feature set is optimal, because recency-only can be stronger on this case and the study does not sweep horizon, sampling rate, or regularization."])
    page(doc, "14. Runtime and Software Cost", ["Modeled AMAT is not wall-clock runtime. The Python prototype performs sampling, feature extraction, delayed-label maintenance, scoring, weight updates, and object bookkeeping. At 32 KiB, the online-to-LRU runtime ratio ranges from 2.71x to 5.78x across the reported cases. No conversion between wall time and synthetic cycles is justified."], image="runtime.png", caption="Figure 4. Measured Python runtime relative to LRU at 32 KiB L3.")
    page(doc, "15. Dashboard Demonstration", ["The local dashboard loads saved benchmark results and exposes workload, capacity, policy, seed, and horizon controls. Recommended demonstration: select Hotset + streaming scans at 16 KiB and compare LRU, SRRIP, and Online in the graph and table.", "Second demonstration: select Changing hotspots at 32 KiB and show that Online can regress. A new single-run demo is useful for interaction testing, but it is not part of the five-seed paper average unless the benchmark is rerun." ])
    page(doc, "16. Validation Tests", ["Automated tests cover hand-calculated hierarchy outcomes, address mapping, set conflicts, exact victim decisions, retained warmup state, endpoint-inclusive labels, censored windows, frozen behavior, pending metadata bounds, trace validation, and counter conservation.", "The tests also verify paired trace hashes and identical upper-cache miss counts across L3 policies. These checks support implementation correctness, not physical-processor validation or external validity."])
    page(doc, "17. Limitations", ["The study uses short synthetic read-only traces, scaled teaching capacities, a serial latency model, and five exploratory seeds. It excludes instruction execution, memory-level parallelism, bandwidth contention, writebacks, coherence, address translation, prefetching, and hardware predictor latency.", "Baselines are conventional rather than a complete state-of-the-art comparison. Hawkeye, DRRIP, and a faithful LeCaR reproduction are not evaluated. The learner has no admission control, phase detector, confidence fallback, or net-cost objective."])
    page(doc, "18. Future Work", ["A stronger follow-up should use licensed application traces, add modern adaptive baselines, and separate development workloads from final evaluation workloads. It should sweep horizon, sample rate, model size, and regularization while reporting the selection protocol.", "Architectural validation should move the selected policy into ChampSim or a comparable simulator and account for predictor storage, inference latency, energy, and queueing. A separate tiered-memory project needs an explicit page migration model."])
    page(doc, "19. Conclusion", ["This project demonstrates a causal online reuse predictor integrated into a reproducible functional cache simulator. Online learning improves selected skewed and scan-polluted workloads, ties on controls, and regresses under changing hotspots.", "The 2.72% mean modeled reduction and 28.98% best reduction show workload-dependent potential, not a universal improvement or hardware speedup. The complete artifact, negative results, ablations, dashboard, and explicit limitations provide a foundation for a larger architectural study."])
    page(doc, "Appendix A. Configuration Snapshot", ["The benchmark uses 64-byte lines, L1/L2/L3 sizes of 32/128/256-1024 lines, associativities 4/8/8, incremental costs 4/12/40/200 cycles, horizon 256, learning rate 0.05, minimum 64 labels, and 1,000-request windows.", "The matrix is 6 workloads x 3 capacities x 5 seeds x 7 policies = 630 simulations. Warmup and measured request counts are retained in the manifest and benchmark JSON."])
    page(doc, "Appendix B. Reproduction Commands", ["Run tests: python -m pytest -q", "Run benchmark: python -m adaptive_memory benchmark --config configs/paper.json", "Regenerate figures and publication report: python -m adaptive_memory report", "Start dashboard: python -m adaptive_memory serve --port 8765", "Generate this editable report: python scripts/create_word_report.py"])
    page(doc, "Appendix C. Data Dictionary", ["amat_cycles is modeled average memory-access time in synthetic cycles. ram_miss_ratio is the final-level miss ratio. cache_hit_ratio summarizes hierarchy hits. fill_bytes counts simplified line-transfer traffic.", "runtime_ms is wall-clock Python execution time and must not be converted into modeled cycles. Positive AMAT reduction means lower AMAT than LRU. Confidence intervals are exploratory percentile bootstrap intervals over paired seeds."])
    page(doc, "Appendix D. Interpretation Checklist", ["Present this as a functional simulation. Say modeled AMAT, not processor latency. Say memory requests, not instructions. Separate the Python runtime graph from cache-cycle graphs.", "Show at least one improvement and one regression. Explain delayed labels and causal information. Explain that a dashboard single-run experiment is not automatically part of the five-seed aggregate."])
    page(doc, "Appendix E. Provenance", ["The saved benchmark completed on Windows 11 with Python 3.13.1. The manifest records source SHA-256 values for simulator, policies, workloads, reporting code, configuration, server, and tests. The source-unchanged-during-run flag is true.", "The report figures are copied from paper/figures and generated from benchmark.json, keeping the embedded graphs aligned with the application artifact."])
    page(doc, "Appendix F. Selected References", ["Jaleel et al., High Performance Cache Replacement Using Re-Reference Interval Prediction, ISCA 2010.", "Jain and Lin, Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement, ISCA 2016.", "Vietri et al., Driving Cache Replacement with ML-based LeCaR, USENIX HotStorage 2018.", "Song et al., Learning Relaxed Belady for Content Distribution Network Caching, USENIX NSDI 2020.", "Yi et al., ArtMem: Adaptive Migration in Reinforcement Learning-Enabled Tiered Memory, ISCA 2025.", "ChampSim project documentation."])
    for sec in doc.sections:
        sec.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        sec.footer.paragraphs[0].text = "Adaptive Memory Lab | Project Report | November 2026"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Created {OUT}")

if __name__ == "__main__":
    build()
