# Changelog

All notable changes to AdapTQ are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Ollama Streaming Generation**: Implemented `generate_streaming()` in `OllamaAdapter` using chunked NDJSON streaming via `/api/generate` with `stream: True`. Resolves issue where Ollama adapter yielded no tokens due to `decode_next()` returning `None`.
- **Ollama Demo Streaming Flag**: Added `--stream` option to `examples/ollama_demo.py` to showcase token streaming in real-time.
- **Unit Tests**: Added offline mock unit test suite in `tests/test_ollama_streaming.py` validating ordered fragment delivery, skipped malformed chunks, HTTP error reporting, and post-exhaustion metric capture.

---

## [0.2.2] — 2026-09-14 — V2.2: Stabilization & Security Baseline

### Security
- **Snapshot Storage Validations**: Added stringent parameter and bound checks to `SessionSnapshot::load()` preventing illegal dimensions, token counts, and bit widths from causing malformed buffer allocations or buffer over-reads. 

### Fixed
- **Max-Lloyd Codebook Boundaries (Issue #16)**: Resolved potential floating point exceptions and undefined behaviors when constructing centroids from vectors containing `NaN`s, massive infinity outliers, or pure-zero sequences.

### Portability
- **AVX2 Dynamic Dispatch (Issue #7)**: Refactored the core SIMD architecture away from global `-mavx2` flags and `#ifdef __AVX2__` guards. The engine now uses `__builtin_cpu_supports` paired with function-specific `#pragma GCC target` attributes. Binary wheels published to PyPI will now safely fall back to scalar processing on older CPUs instead of crashing with `SIGILL`.

### Tests
- **Boundary Condition Regressions**: Extended C API tests to exhaustively validate `adaptq_append` edge cases (NaNs, infinite tensors).
- **Capability Testing**: CTest suite automatically accommodates the AVX2 capability detection framework.

---

## [0.2.1] — 2026-07-22 — V2.1: Real Runtime Integration & Validation

### Added

#### Python — Multi-Backend Adapter Layer
- **`adaptq/runtime_py/`** — `IRuntimeAdapter` Python ABC + backend registry
  - `create_adapter(backend)` factory for all supported backends
  - `backend_available(backend)` / `list_available_backends()` discovery helpers
  - `IRuntimeAdapter.generate_streaming()`: Added metrics tracking (wall time, tokens/sec, token IDs, KV stats) and lifecycle guarantees via `generation_result()`
- **`adaptq/runtime_py/backends/transformers_hf.py`** — HuggingFace Transformers adapter (Priority 1)
  - Hooks into `DynamicCache.update()` to intercept K/V per layer
  - Supports any CausalLM: Qwen, LLaMA, Mistral, Gemma, GPT-2, …
  - CPU and GPU; auto-detects device
- **`adaptq/runtime_py/backends/llama_cpp_python.py`** — llama-cpp-python adapter (Priority 1)
  - Hooks into `Llama.eval()` + `Llama.sample()` token loop
  - Full generate() pipeline with save/restore llama context state
  - KV stats via size estimation (approximate)
- **`adaptq/runtime_py/backends/ollama.py`** — Ollama REST adapter (Priority 2)
  - POST /api/generate with streaming JSON
  - No internal KV access; captures timing and token counts
- **`adaptq/runtime_py/backends/vllm.py`** — vLLM stub (Priority 3, CUDA required)
- **`adaptq/runtime_py/backends/mlx.py`** — MLX stub (Priority 5, Apple Silicon)
- **`adaptq/runtime_py/backends/llama_cpp.py`** — llama.cpp C++ stub (Priority 6, source build)

#### C++ — IRuntimeAdapter Interface
- **`runtime/adapters/adapter.h`** — `IRuntimeAdapter` pure C++ interface (9 virtual methods)
- **`runtime/adapters/runtime_metadata.h`** — `ModelConfig`, `SessionConfig`, `KVCacheView`, `RuntimeMetadata`, `GenerationResult`

#### Examples (6 demos)
- `examples/transformers_demo.py` — full Transformers pipeline demo
- `examples/llama_demo.py` — full llama-cpp-python pipeline + context save/restore
- `examples/ollama_demo.py` — Ollama REST generation demo
- `examples/snapshot_demo.py` — standalone snapshot pipeline (no model required)
- `examples/branch_replay_demo.py` — branch replay at midpoint
- `examples/compare_backends.py` — cross-backend comparison table

#### Benchmarks
- `benchmarks/bench_all.py` — unified benchmark runner (KV memory, latency, snapshot size)
- Output formats: JSON, CSV, Markdown

#### Integration Tests
- `integration_tests/conftest.py` — shared fixtures, markers, `--model-hf`, `--model-gguf` CLI args
- `integration_tests/test_adapter_contract.py` — 25 backend-agnostic contract tests
- `integration_tests/test_transformers.py` — 15 end-to-end transformers tests
- `integration_tests/test_llama_cpp_python.py` — 10 end-to-end llama-cpp-python tests

#### CI
- `.github/workflows/integration.yml` — 4-job CI: C++ tests, adapter contracts, transformers integration, Python validation

#### Documentation
- `docs/v2.1_integration.md` — backend integration guide + "how to add a new backend"

### Changed
- `adaptq/__init__.py` — exports `create_adapter()`, `list_available_backends()`
- `pyproject.toml` — version `0.2.1`; optional deps for each backend
- `pyproject.toml` — `[tool.pytest.ini_options]` now covers `integration_tests/`
- `CLAUDE.md` — V2.1 architecture notes

### Fixed (Pre-Release Audit)
- **CRITICAL**: Fixed `SyntaxError` in `adaptq/__init__.py` (duplicate docstring) that prevented the package from being imported.
- Fixed `adaptq/__init__.py` missing `__version__`.
- Fixed `adaptq/replay.py` stale binary path (`build_v2` → `build_release`), duplicate `--format` flag, and namespace pollution.
- Fixed `adaptq/runtime_py/backends/transformers_hf.py` `torch_dtype` deprecation warning (changed to `dtype`).
- Expanded `.gitignore` to cover all build directories (`build_release`, `build_debug`, etc.).
- Made `validate_*.sh`, `replay.py`, and `run_tests.py` fully cross-platform (removed hardcoded WSL paths).

### Test Results (V2.1)
- **C++**: 67/67 tests pass (unchanged)
- **Integration contract**: 25/25 pass (no model required)
- **Transformers integration**: 15/15 pass (Qwen2-0.5B, CPU, 167s)
- **Backends available**: `transformers`, `llama_cpp_python`, `ollama`

---

## [0.2.0] — 2026-07-22 — V2: Replay + Compare


### Added

#### C++ (Core)
- **`runtime/runtime_context.h/cpp`** — `RuntimeContext` orchestration class
  - Full wire-up: `IPolicy → IKVStrategy → IStorageBackend → IKernelBackend`
  - Token log (`log_tokens=true`) for FP32 K/V capture used by snapshot replay
  - `StrategyFactory` / `StorageFactory` typedefs; runtime strategy registry
  - `strategy_factory_by_name()` and `strategy_names()` registry functions
  - `make_contiguous()` public factory for `ContiguousSlabStorage`
- **`replay/session_snapshot.h/cpp`** — `SessionSnapshot` versioned binary format
  - Magic `0x41515353` ("AQSS"), version 2, little-endian self-describing header
  - Per-head K/V slab data, scales, format tags capture
  - Optional strategy state serialization via `IReplayHooks`
  - Optional token log for deterministic replay
  - `capture()`, `save()`, `load()` API
- **`replay/replay_engine.h/cpp`** — `ReplayEngine`
  - `replay()` — full deterministic replay from token 0
  - `branch(from_token)` — warm-up replay then return ready context
  - `replay_with(strategy_name)` — cross-strategy replay
  - `ReplayReport` with per-token `ComputeMetrics`
- **`cli/cmd_replay.cpp`** — `adaptq replay <snapshot.aqss> [options]`
  - `--strategy har_fixed|fp_passthrough` — override strategy
  - `--from-token N` — branch mode
  - `--metrics` — collect per-token metrics
  - `--output <file>` — file output
  - `--format json|csv|md|tex` — output format
- **`cli/cmd_compare.cpp`** — `adaptq compare <snapshot.aqss> --strategies A,B`
  - JSON/CSV/Markdown/LaTeX output
  - Aggregated per-strategy summary: wall time, avg latency, avg quality, avg bits/dim
- **`cli/cmd_create_strategy.cpp`** — `adaptq create-strategy <Name>`
  - Scaffolds full `IKVStrategy` directory under `strategies/<name>/`

#### Tests (C++)
- `tests/unit/test_runtime_context.cpp` — 10 tests for RuntimeContext
- `tests/unit/test_session_snapshot.cpp` — 8 tests for SessionSnapshot
- `tests/unit/test_replay_engine.cpp` — 9 tests for ReplayEngine

#### Python
- `adaptq/replay.py` — `ReplayEngine`, `ReplayResult`, `CompareResult`, `snapshot_info`
  - Thin subprocess wrappers around `adaptq replay`/`compare` CLI
  - `CompareResult.to_markdown()`, `best_quality()`, `fastest()`

### Changed
- `adaptq/__init__.py` — exports `ReplayEngine`, `ReplayResult`, `CompareResult`, `snapshot_info`
- `pyproject.toml` — version `0.2.0`, Python `>=3.8`, proper classifiers and extras
- `README.md` — full V2 documentation with replay/compare usage, CLI reference, snapshot format, changelog

### Fixed
- `cli/cmd_create_strategy.cpp` — missing `adaptq::` namespace wrapper caused linker error
- `replay/session_snapshot.cpp` — `cache_size` was set from `csb->size()` (total K+V slots) instead of `csb->size()/2` (K/V pairs)
- `tests/unit/test_session_snapshot.cpp` — non-copyable `RuntimeContext` returned by value; fixed to `unique_ptr`; added missing `#include <fstream>`
- `tests/unit/test_runtime_context.cpp` — inline `extern IStorageBackend *make_contiguous()` declaration inside lambda didn't resolve `adaptq::` namespace; fixed to `adaptq::make_contiguous()`

### Test Results
- **C++**: 67/67 tests pass (V1: 38, V2: 29)
- **Python**: 5/5 validation stages pass

---

## [0.1.0] — Initial Release — V1: Plugin Architecture

### Added
- **Plugin interfaces**: `IKVStrategy`, `IPolicy`, `IStorageBackend`, `IKernelBackend`, `ICostFunction`, `IQualityOracle`
- **Strategies**: `HARFixedStrategy` (4-bit HAR quantization), `FPPassthroughStrategy` (FP32 passthrough)
- **Policy**: `UniformPolicy`
- **Storage backends**: `ContiguousSlabStorage`, `SegmentedSlabStorage`
- **Quality oracle**: default ensemble oracle
- **Cost optimizer**: `ICostFunction`-based budget management
- **C API**: `adaptq_create`, `adaptq_append`, `adaptq_compute`, `adaptq_destroy`, MHA context
- **Python bindings**: pybind11 `MHAContext` → `adaptq.Engine`
- **PyTorch adapter**: `AdaptQAttention` drop-in module
- **38/38 C++ tests** pass
- **5/5 Python validation stages** pass

[0.2.0]: https://github.com/l3tchupkt/adaptq/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/l3tchupkt/adaptq/releases/tag/v0.1.0
