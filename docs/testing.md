# AdapTQ Testing Guide

## Overview

AdapTQ has three test layers that together validate the full stack:

| Layer | Runner | What it covers |
|-------|--------|----------------|
| C++ unit tests (V1) | `ctest` | Algorithms, storage, C ABI |
| C++ conformance tests | `ctest` | `IKVStrategy` contract |
| C++ unit tests (V2) | `ctest` | RuntimeContext, SessionSnapshot, ReplayEngine |
| Python validation suite | `python tests/run_tests.py` | MSE calibration, benchmarks, ctypes C ABI, cross-validation |

**Total: 67 C++ tests (V1: 38 + V2: 29) + 5 Python stages.**

---

## Running the Tests

### Linux / WSL (full suite)

```bash
cd adapTQ
cmake -B build_v2 -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2 --parallel 4
cd build_v2 && ctest --output-on-failure -j4
```

### Windows (MSVC + Ninja) — from VS Developer Command Prompt

```cmd
cd adapTQ
cmake -B build_v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl
cmake --build build_v2 --parallel
cd build_v2 && ctest --output-on-failure
```

### Python validation suite (Windows or Linux)

```bash
python tests/run_tests.py          # all 5 stages
python tests/run_tests.py --stage 2  # single stage
```

Stages:

| # | Stage | What it proves |
|---|-------|----------------|
| 1 | C++ ctest via WSL | 67 C++ tests pass (V1: 38, V2: 29) |
| 2 | MSE calibration verification | Threshold changes are justified |
| 3 | Python reference benchmark | Throughput and attention correctness |
| 4 | C ABI integration (ctypes) | C++ library callable from Python |
| 5 | Cross-validation | C++ cosine-similar to FP32 reference |

## V2 Test Descriptions

### RuntimeContext (10 tests)

| Test | What it verifies |
|------|-----------------|
| `init does not throw` | Basic construction with config |
| `get_strategy / get_storage return non-null` | Per-head initialization |
| `append increases storage usage` | Hot-path write path |
| `compute returns valid ComputeMetrics after append` | Full compute path |
| `compute on empty cache returns zero output` | Edge case: empty cache |
| `reset clears storage and resets token_pos` | Session reset |
| `multiple appends accumulate` | 10-token accumulation |
| `fp_passthrough strategy init` | Alternative strategy factory |
| `strategy_factory_by_name registry` | Name-to-factory lookup |
| `token log is populated when log_tokens=true` | Snapshot prerequisite |

### SessionSnapshot (8 tests)

| Test | What it verifies |
|------|-----------------|
| `capture from RuntimeContext is valid` | Basic capture |
| `capture without token log has no log` | Minimal capture mode |
| `capture with token log contains entries` | Token log correctness |
| `save and load roundtrip` | Binary I/O fidelity |
| `load rejects bad magic` | Format guard |
| `load missing file throws` | Error handling |
| `strategy state captured when IReplayHooks available` | IReplayHooks integration |
| `head snapshot has valid storage data` | Storage data size correctness |
| `save/load is deterministic for same input` | Bit-exact reproducibility |

### ReplayEngine (9 tests)

| Test | What it verifies |
|------|-----------------|
| `full replay completes without error` | Happy path |
| `full replay produces non-empty report` | Metrics collection |
| `replay without token log throws` | Missing log guard |
| `branch at 0 leaves context empty` | Branch at start |
| `branch at N puts N tokens in storage` | Branch warm-up |
| `branch at from_token > n_tokens throws` | Out-of-range guard |
| `cross-strategy replay with fp_passthrough` | Strategy override |
| `cross-strategy replay unknown strategy throws` | Registry guard |
| `replay strategy name matches context strategy` | Report metadata |
| `full replay and branch produce same storage at branch point` | Determinism |



---

## Why Are the MSE Thresholds "High"?

The `Quantizer` is evaluated after the full **HAR pipeline**:

```
Input x  (N(0,1), dim=128)
  │
  ▼  L2-normalize   →   x̂ = x / ‖x‖
  │
  ▼  Rademacher D  →   x̂ ⊙ D
  │
  ▼  FWHT           →   ŷ = H(D x̂)
  │
  ▼  ±3σ soft-clip  →   clip ŷ to [−3σ, 3σ], map to [−1, 1]
  │
  ▼  Vector quantization (2, 3, or 4 bit)
  │
  ▼  Inverse FWHT + D + rescale by (‖x‖ × clip)
  │
  ▼  x̂_reconstructed
```

The **reconstruction MSE is measured vs the original raw input `x`**, not vs the
L2-normalized version. This means the error budget includes:

- The VQ approximation error (depends on bits)
- The ±3σ clipping error (tails that exceeded the codebook range)
- Normalization scale compounding: `scale = ‖x‖ × clip`, so for N(0,1)
  input of dimension 128, `‖x‖ ≈ 11.3` and `clip = 3σ ≈ 3.0`.

### Calibrated thresholds

| bits | Observed MSE | Threshold | Headroom | What fails |
|------|-------------|-----------|----------|------------|
| 4    | ~0.057      | 0.15      | 2.6×     | If VQ table corrupt |
| 3    | ~0.192      | 0.50      | 2.6×     | If FWHT wrong |
| 2    | ~0.678      | 1.50      | 2.2×     | If packing wrong |

### Why the original thresholds were wrong

The original thresholds (0.05 / 0.10 / 0.20) were calibrated against **unit
vectors** (as used in `python/benchmark.py`'s `rand_unit()` helper). For unit
vectors, `‖x‖ = 1`, so `scale = 1 × clip ≈ 3`, and VQ error in the
normalised domain maps back to small raw-space error.

The C++ test uses **unnormalised N(0,1) vectors** where `‖x‖ ≈ √128 ≈ 11.3`,
so the raw-space MSE is correspondingly larger. The thresholds were updated
after measuring actual algorithm output (not estimated from theory).

**Important:** these thresholds do NOT represent degraded quality. The
attention cosine similarity between the C++ quantised output and the FP32
reference remains `≥ 0.99` at 4-bit and `≥ 0.90` at 2-bit across all tested
configurations (Stage 5 cross-validation).

---

## Codebook Reference

Max-Lloyd optimal codebooks for the three precision levels, tuned for the
Rademacher-FWHT distribution after ±3σ normalisation (values in [−1, 1]):

| bits | Levels | Codebook |
|------|--------|---------|
| 2    | 4      | `−1.5104, −0.4528, 0.4528, 1.5104` |
| 3    | 8      | `−2.1529 … +2.1529` (8 values) |
| 4    | 16     | `−2.7326 … +3.5714` (16 values, asymmetric) |

The asymmetric 4-bit codebook is intentional: the FWHT output has a slightly
positive-skewed tail distribution after Rademacher mixing, so more centroid
density is allocated to positive values.

---

## Adding a New Strategy

A new `IKVStrategy` implementation automatically gets the full conformance
suite via `run_conformance<T>()`:

```cpp
// tests/conformance/strategy_conformance.cpp
#include "../../strategies/my_strategy.cpp"

TEST_CASE("MyStrategy conformance", "[conformance]") {
    adaptq::run_conformance<adaptq::MyStrategy>("MyStrategy");
}
```

The harness tests all six sub-contracts:
1. Lifecycle (`init`, `reset`)
2. `ICompression::compress()` writes to storage
3. `ICompression::decompress()` produces non-zero output
4. `IEviction::on_append()` triggers eviction at capacity
5. `IEviction::on_attention()` does not crash
6. Optional `IQuality` and `IReplayHooks` sub-contracts

---

## CI Matrix

See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) for the full
matrix. Summary:

```
Ubuntu-latest
  CMake -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build
  ctest --output-on-failure
  python tests/run_tests.py        ← all 5 stages

Windows-latest
  ilammy/msvc-dev-cmd (VS2022 x64)
  CMake -G Ninja -DCMAKE_CXX_COMPILER=cl
  cmake --build
  ctest --output-on-failure
  python tests/run_tests.py --stage 2+3
```
