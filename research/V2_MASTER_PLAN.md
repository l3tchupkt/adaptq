# AdapTQ V2: Adaptive KV-Cache Runtime Master Plan

## 1. Executive Summary
AdapTQ is evolving from a fixed-precision KV-cache quantizer into an industry-grade, research-oriented adaptive KV-cache runtime. The V2 architecture will intelligently manage memory budgets, assigning different quantization precisions (FP16, INT8, INT4, INT2) to cache pages based on their estimated importance to generation quality. This allows accommodating significantly larger context sizes within a fixed hardware memory budget while minimizing generation degradation. The project shifts to a page-oriented storage model, separates core algorithms from runtime policy engines, introduces importance-aware eviction/demotion, and provides asynchronous background compression.

## 2. Current Architecture
- **Core Algorithms:** AVX2/Scalar compressed-domain dot product, Fast Walsh-Hadamard Transform (FWHT), Max-Lloyd codebooks.
- **Data Layout:** Giant contiguous ring-buffer cache allocations (e.g., `AttentionHead` pre-allocates for maximum capacity), single fixed precision across the entire cache.
- **Storage Model:** `SegmentedSlabStorage` providing fixed slot-based slab tracking.
- **C API / Integrations:** Single-head and Multi-head C API (`adaptq_c_api.cpp`), wrapped in Python bindings via `ctypes`. Experimental integration with `llama.cpp` and standalone C testers exist.
- **State Management:** Snapshots via `SessionSnapshot`, and replay verification via `ReplayEngine`.

## 3. Problems (Current Bottlenecks & Limitations)
- **Monolithic Caches:** The cache is allocated statically upon initialization. It cannot dynamically page memory out, page memory in, or efficiently share prefixes across multiple independent generations.
- **Fixed Precision:** Every token in the cache uses the exact same bit-width, squandering memory on unimportant tokens that could tolerate heavy compression (INT2), and losing quality on important tokens (FP16).
- **Synchronous Compression:** Token quantization runs directly on the critical inference path.
- **Lack of Memory Budgeting:** The system relies on "capacity limits" rather than real-world "memory budgets".
- **Lack of Importance Awareness:** No concept of which KV tokens actually impact the attention scores the most.

## 4. Proposed V2 Architecture
```text
                    LLM / Framework (PyTorch, Transformers, vLLM)
                                      |
                             Unified KV API
                                      |
                     +--------------------------------+
                     |       Adaptive Runtime         |
                     | - Policy Engine                |
                     | - Scheduler                    |
                     | - Memory Manager               |
                     +--------------------------------+
                                      |
             +------------------------+-----------------------+
             |                        |                       |
             v                        v                       v
          HOT KV (FP16)            WARM KV (INT4)          COLD KV (INT2)
             |                        |                       |
             +------------------------+-----------------------+
                                      |
                             Paged KV Storage
                                      |
            +-------------------------+-------------------------+
            v                                                   v
 Asynchronous Compression Engine                    Compressed Attention Engine
            |                                                   |
            +--< AVX2 | AVX512 | NEON | CUDA | Scalar Fallback >--+
```
- **Runtime:** Handles memory budgets, policies, scheduling, and precision migration.
- **Storage:** Paged metadata (`KVPage`) allowing non-contiguous allocation and shared prefixes.
- **Policy Engine:** Dynamically assigns precisions (e.g., `RecencyPolicy`, `AttentionHistoryPolicy`).
- **Telemetry:** Exposes real-time stats on cache distribution and memory pressure.

## 5. Research Hypotheses
- **RQ1:** Adaptive precision outperforms fixed-bit KV quantization at the same memory budget.
- **RQ2:** Importance-aware allocation reduces downstream quality degradation compared to naive eviction.
- **RQ3:** Multi-tier KV storage can reduce memory footprint without increasing latency significantly.
- **RQ4:** Asynchronous compression can completely remove compression overhead from the critical inference path.
- **RQ5:** Query-aware page selection preserves quality while drastically reducing attention FLOPs.

## 6. Baseline Metrics & Test Results
**Environment:** Linux (WSL), Python 3.13.12, PyTorch/Transformers (CPU).
**Tests:** 
- Native CTest Suite (78/78 passing)
- Pytest Suite (9/9 passing)
- No AVX2/Scalar API memory violations (Stack buffer overflows resolved).

**Benchmark Profile (Transformers Integration, Llama/Mistral dummy model):**
- **Generation Latency:** 3.74 tokens/sec (267.39 ms/token latency)
- **KV Memory Compression:** 4.0x compression ratio vs FP16 at 4-bits.
- **Memory Footprint:** 1024 tokens @ 4-bits = 1MB (adaptq) vs 4MB (FP16). 
- **Snapshot Size:** ~768 KB for 512 tokens at 4-bits.

## 7. Phase Plan
- **Phase 0:** Repository Forensics & Audit (Completed via this document).
- **Phase 1:** Freeze Baseline (Completed via `bench_all.py` suite).
- **Phase 2:** Runtime Abstraction (Move monolithic classes into `cache_manager` and `policy`).
- **Phase 3:** Paged KV Cache (Replace contiguous array allocations with `KVPage`).
- **Phase 4:** Multi-Precision KV Tiers (Support INT2/3/4/8/FP16 simultaneously).
- **Phase 5:** Importance-Aware Compression (Implement algorithms tracking token attention hits).
- **Phase 6:** Memory-Budget-Aware Runtime (Specify budgets like `4GB` rather than `capacity`).
- **Phase 7:** Quality-Constrained Optimization.
- **Phase 8:** Asynchronous Compression (Background workers).
- **Phases 9-14:** Query-Aware Selection, Compressed Attention, HW Backends, Batching, Prefix Sharing, Offloading.
- **Phases 15-37:** Productionization (Snapshots, Telemetry, Framework Integrations, CI, Paper Writing).

## 8. Experiments & Ablations
Each Phase introduces an experiment folder (e.g., `experiments/002_paging/`) tracking:
- Target configuration
- MSE, Cosine Similarity, Perplexity
- Latency (p50/p95/p99)
- Effective bits/token

## 9. Risk Register
1. **Performance regressions:** Abstraction layers (e.g. paging) might destroy cache locality for AVX2 instructions. *Mitigation: Benchmark LUT attention traversal heavily.*
2. **Metadata bloat:** Paging and mixed-precision tracking could consume more memory than the quantization saves. *Mitigation: Track theoretical vs actual compression ratio meticulously.*
3. **Thread-safety:** Async compression introduces race conditions. *Mitigation: Strict locking, lock-free queues, and TSAN integration.*

## 10. Compatibility Strategy
- **Baseline Algorithm:** The existing `AttentionHead` and quantizer will NOT be deleted. They act as the FP16/INT4 baselines to compare against.
- **C ABI:** Struct layouts will be maintained where possible, or appropriately versioned. 

## 11. Testing Strategy
- Continuous validation via CTest, Pytest.
- Golden reference testing (Scalar vs AVX2 outputs).
- Fuzzing of C APIs, snapshot loaders, and malformed inputs.
- Real-model evaluations (Perplexity, Needle-in-a-Haystack) instead of just MSE.

## 12. Security Strategy
- Reject untrusted serialized inputs immediately.
- Strict bounds checking (addressing `STATUS_STACK_BUFFER_OVERRUN` risks proactively).
- ASAN and UBSAN enforcement in CI pipelines.

## 13. Integration Strategy
- **Transformers:** Provide a clean `adaptq.cache` object complying with HuggingFace cache spec.
- **PyTorch:** Expose a native `torch.nn.Module` to minimize Python<->C++ crossing.
- **vLLM/llama.cpp:** Implement thin wrappers adhering strictly to their pagination rules.

## 14. Release Strategy
Iterative releases.
- **v2.1:** Runtime abstraction and paging.
- **v2.2:** Multi-precision & importance-aware management.
- **v2.3:** Asynchronous workers & telemetry.
- **v2.4:** Framework adapters (Transformers, PyTorch).
