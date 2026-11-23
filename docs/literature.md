# Related Work and Research Positioning

Sources were checked online during project preparation. These papers establish prior work; their performance numbers are not results of this project.

| Work | What it studies | Relationship to this project |
| --- | --- | --- |
| [Jaleel et al., RRIP, ISCA 2010](https://jaleels.org/ajaleel/publications/isca2010-rrip.pdf) | Cache replacement using re-reference interval prediction | Source for the implemented two-bit SRRIP baseline. DRRIP is not implemented. |
| [Jain and Lin, Hawkeye, ISCA 2016](https://class.ece.iastate.edu/tyagi/cpre581/papers/ISCA16Belady.pdf) | Learning replacement behavior from Belady's decisions on past accesses | Important CPU-cache prior work. Our logistic predictor does not reproduce Hawkeye and is not compared experimentally against it. |
| [Vietri et al., LeCaR, HotStorage 2018](https://www.usenix.org/conference/hotstorage18/presentation/vietri) | Regret minimization combining LRU and LFU | Shows that learning between cache strategies is established. This project predicts reuse directly instead. |
| [Song et al., Learning Relaxed Belady, NSDI 2020](https://www.cs.princeton.edu/~wlloyd/papers/lrb-nsdi20.pdf) | Learned replacement in CDN object caches | Provides background on learned caching, with a different workload and object model. |
| [Yi et al., ArtMem, ISCA 2025](https://doi.org/10.1145/3695053.3731001) | Reinforcement learning for tiered-memory page migration | Relevant to the original broad data-placement idea; a different control layer and granularity. |
| [ChampSim documentation](https://champsim.github.io/ChampSim/develop/) | Trace-based architectural simulation | A possible future route to stronger CPU-performance evaluation. No ChampSim integration is included. |

The broad claim "AI for intelligent memory placement" is not a new research contribution by itself. The present artifact supports an educational study of delayed online learning, workload shifts, feature ablations, and overhead. A publication claim needs stronger differentiation and empirical comparison. No claim of a comprehensive or exhaustive literature review is made.

