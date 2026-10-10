#pragma once
#include <cstdint>

/* -------------------------------------------------------------------------
 * adaptq/config.h — HeadConfig and RuntimeConfig structs
 *
 * These are plain-data configuration structs passed to IKVStrategy::init()
 * and IPolicy::init(). They are intentionally simple so they can be
 * constructed inline without any heap allocation.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/** Per-head configuration passed to IKVStrategy::init(). */
struct HeadConfig {
    int      dim      = 128;    /* head dimension                           */
    int      bits     = 4;      /* quantization bits: 2, 3, or 4           */
    int      capacity = 4096;   /* max tokens in the KV cache ring buffer   */
    uint64_t seed     = 0;      /* Rademacher seed (unique per layer+head)  */
    float    v_mass   = 0.f;    /* sparse-V threshold (0 = full V)          */
    int      hybrid_thresh = 0; /* FP32 path for n_tokens < this (0 = off) */
};

/** Runtime-wide configuration passed to IPolicy::init(). */
struct RuntimeConfig {
    int   n_layers    = 1;
    int   n_heads     = 1;
    int   dim         = 128;
    int   bits        = 4;
    int   capacity    = 4096;
    float v_mass      = 0.f;
    float memory_budget_mb  = 2048.f;
    float quality_floor     = 0.f;
    float latency_hard_limit_us = 0.f;  /* 0 = no limit */
};

} /* namespace adaptq */
