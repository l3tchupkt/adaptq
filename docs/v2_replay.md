# AdapTQ V2: Replay + Compare

## Overview

AdapTQ V2 introduces deterministic **session replay** and **cross-strategy comparison** — core infrastructure for offline analysis, regression testing, and strategy evaluation.

The V2 execution path:

```
RuntimeContext
  → IPolicy (strategy selection per token)
  → IKVStrategy (compression + eviction per head)
      → ICompression (quantize K/V)
      → IEviction (evict tokens)
  → IStorageBackend (raw slot storage)
  → IKernelBackend (FWHT + kdot + vaccum)
  → IQualityOracle (quality estimation)
```

---

## SessionSnapshot (.aqss)

A `.aqss` (AdapTQ Session Snapshot) file is a versioned binary capture of a `RuntimeContext` at any token position.

### Binary Format

```
Header (40 bytes):
  [4B]  magic     = 0x41515353  ("AQSS")
  [4B]  version   = 2
  [4B]  n_layers
  [4B]  n_heads
  [4B]  dim
  [4B]  bits
  [4B]  n_tokens     (token count at capture time)
  [4B]  n_heads_total (n_layers × n_heads)
  [8B]  flags        (bit 0 = has_token_log, bit 1 = has_strategy_state)

Per-head blocks (n_heads_total entries):
  [4B]  layer
  [4B]  head
  [4B]  cache_size   (number of K/V pairs)
  [4B]  slot_bytes
  [8B]  data_bytes
  [N]   storage_data (K+V slab, little-endian)
  [4B]  n_scales
  [N]   scales       (float32 per slot)
  [4B]  n_tags
  [N]   format_tags  (uint8 per slot)

Optional — Strategy state (flags & 2):
  Per-head:
    [4B]  state_len
    [N]   state_blob (IReplayHooks::serialize_state output)

Optional — Token log (flags & 1):
  [4B]  n_log_entries
  Per-entry:
    [4B]  layer
    [4B]  head
    [4B]  dim
    [dim*4B]  k_fp32
    [dim*4B]  v_fp32
```

**Properties:**
- Little-endian throughout
- Self-describing: no external schema required
- Forward-compatible: version guard on `load()`
- Token log is optional but required for replay

### Capturing a Snapshot (C++)

```cpp
#include "runtime/runtime_context.h"
#include "replay/session_snapshot.h"

// Enable token logging before running inference
RuntimeContextConfig cfg;
cfg.n_layers   = 4;
cfg.n_heads    = 8;
cfg.dim        = 128;
cfg.log_tokens = true;  // ← required for replay

RuntimeContext ctx;
ctx.init(cfg);

// Run inference...
for (int t = 0; t < n_tokens; ++t) {
    for (int l = 0; l < 4; ++l)
        for (int h = 0; h < 8; ++h)
            ctx.append(l, h, k_data[t][l][h], v_data[t][l][h]);
}

// Capture and save
auto snap = SessionSnapshot::capture(ctx, /*include_token_log=*/true);
snap.save("session.aqss");
```

---

## ReplayEngine

### Full Replay

Re-feed all tokens from the snapshot into a fresh `RuntimeContext`:

```cpp
#include "replay/replay_engine.h"

SessionSnapshot snap = SessionSnapshot::load("session.aqss");

RuntimeContextConfig cfg;
cfg.n_layers = snap.n_layers();
cfg.n_heads  = snap.n_heads();
cfg.dim      = snap.dim();
cfg.bits     = snap.bits();

RuntimeContext ctx;
ctx.init(cfg);  // default strategy (har_fixed)

ReplayEngine engine(/*collect_metrics=*/true);
ReplayReport report = engine.replay(snap, ctx);

printf("Replayed %d tokens in %.2f ms\n",
       report.n_tokens_replayed, report.wall_time_ms);
```

### Branch Replay

Warm-up to a specific token, then continue inference:

```cpp
ReplayEngine engine;
engine.branch(snap, ctx, /*from_token=*/128);
// ctx is now ready for continued inference from token 128
```

### Cross-Strategy Replay

Replay the same token log with a different strategy:

```cpp
ReplayEngine engine(/*collect_metrics=*/true);
ReplayReport report = engine.replay_with(
    snap, "fp_passthrough", cfg);

printf("fp_passthrough: avg quality = %.4f\n",
       /* compute from report.metrics */);
```

---

## CLI Reference

### `adaptq replay`

```
Usage: adaptq replay <snapshot.aqss> [options]

Options:
  --strategy har_fixed|fp_passthrough   Override strategy
  --from-token N                         Branch at token N
  --metrics                              Collect per-token metrics
  --output <file>                        Write output to file
  --format json|csv|md|tex               Output format (default: json)
```

**Example — Full replay with metrics:**
```bash
./build_v2/adapTQ_demo replay session.aqss --metrics --format md
```

**Example — Branch at token 256:**
```bash
./build_v2/adapTQ_demo replay session.aqss --from-token 256
```

**JSON output format:**
```json
{
  "strategy": "har_fixed",
  "n_tokens_replayed": 512,
  "wall_time_ms": 8.3,
  "snapshot_n_tokens": 512,
  "n_layers": 4,
  "n_heads": 8,
  "dim": 128,
  "metrics": [
    {"layer": 0, "head": 0, "n_tokens_used": 1, "latency_us": 0.2,
     "quality": 0.97, "avg_bits_per_dim": 4.0},
    ...
  ]
}
```

---

### `adaptq compare`

```
Usage: adaptq compare <snapshot.aqss> --strategies A,B[,C] [options]

Options:
  --strategies har_fixed,fp_passthrough  (required) Comma-separated list
  --format json|csv|md|tex               Output format (default: json)
  --output <file>                        Write output to file
```

**Example — Compare two strategies:**
```bash
./build_v2/adapTQ_demo compare session.aqss \
  --strategies har_fixed,fp_passthrough \
  --format md
```

**Markdown output:**
```markdown
# Strategy Comparison

| Strategy | Tokens | Wall (ms) | Latency µs | Quality | Bits/Dim |
|---|---|---|---|---|---|
| `har_fixed` | 512 | 8.30 | 0.08 | 0.9700 | 4.00 |
| `fp_passthrough` | 512 | 12.45 | 0.12 | 1.0000 | 32.00 |
```

**CSV output:**
```csv
strategy,n_tokens,wall_time_ms,avg_latency_us,avg_quality,avg_bits_per_dim
har_fixed,512,8.30,0.08,0.97,4.00
fp_passthrough,512,12.45,0.12,1.00,32.00
```

**LaTeX output:**
Generates a `\begin{table}` block ready for inclusion in papers.

---

## Python API

```python
from adaptq import ReplayEngine, snapshot_info

# Quick metadata check
info = snapshot_info("session.aqss")
# → {'n_tokens': 512, 'n_layers': 4, 'n_heads': 8, 'dim': 128}

engine = ReplayEngine()

# Full replay
result = engine.replay("session.aqss", collect_metrics=True)
print(result.strategy, result.n_tokens_replayed, result.wall_time_ms)

# Branch at token 256
result = engine.replay("session.aqss", from_token=256)

# Override strategy
result = engine.replay("session.aqss", strategy="fp_passthrough")

# Cross-strategy comparison
compare = engine.compare(
    "session.aqss",
    strategies=["har_fixed", "fp_passthrough"]
)

# Inspect results
print(compare.best_quality())   # row with highest avg_quality
print(compare.fastest())        # row with lowest wall_time_ms
print(compare.to_markdown())    # Markdown table
```

---

## IReplayHooks — Strategy State Serialization

Strategies that implement `IReplayHooks` have their internal state serialized into the snapshot. This enables exact state restoration after `branch()`:

```cpp
struct IReplayHooks {
    virtual void serialize_state(std::ostream &out) const = 0;
    virtual void deserialize_state(std::istream &in) = 0;
};
```

`HARFixedStrategy` implements `IReplayHooks`. If your custom strategy maintains internal state (importance counters, EMA buffers, etc.), implement this interface to make it branch-compatible.

---

## Architectural Notes

### Storage Layout
`ContiguousSlabStorage` is initialized with `capacity * 2` slots: slots `[0, cache_sz)` hold K vectors; slots `[cache_sz, 2*cache_sz)` hold V vectors. The snapshot `cache_size` field stores the number of K/V **pairs** (not total slots).

### Token Log Layout
Token log entries are ordered: for token `T`, entries occupy indices `[T * n_layers * n_heads, (T+1) * n_layers * n_heads)`, with order `(l=0,h=0), (l=0,h=1), ..., (l=L-1,h=H-1)`.

### Thread Safety
`RuntimeContext` is NOT thread-safe. Use one instance per thread. `ReplayEngine` is stateless and can be shared.
