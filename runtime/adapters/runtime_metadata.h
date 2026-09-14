#pragma once
#include <cstdint>
#include <string>
#include <vector>

/* =========================================================================
 * runtime/adapters/runtime_metadata.h
 *
 * Plain-data structs shared by IRuntimeAdapter and all backend adapters.
 * No engine-specific types — float*, int32_t*, std::string only.
 * ======================================================================= */

namespace adaptq {

/* ---- Model configuration ------------------------------------------------ */
struct ModelConfig {
    std::string model_path;       /* Path to model weights (GGUF, HF dir, etc.) */
    std::string tokenizer_path;   /* Tokenizer path (same as model_path for HF) */
    int         n_ctx      = 4096;/* Context window size */
    int         n_threads  = 4;   /* CPU threads for inference */
    int         n_gpu_layers = 0; /* GPU layers to offload (0 = CPU only) */
    float       temperature = 0.8f;
    float       top_p       = 0.95f;
    int         seed        = 42;
};

/* ---- Session configuration ---------------------------------------------- */
struct SessionConfig {
    std::string prompt;
    int         max_new_tokens = 128;
    bool        log_tokens     = true;  /* Enable KV log for snapshot */
    bool        stream         = false; /* Stream tokens as they are generated */
};

/* ---- KV cache view (non-owning) ----------------------------------------- */
struct KVCacheView {
    int          n_layers;
    int          n_heads;
    int          head_dim;
    int          n_tokens;     /* Number of tokens currently cached */
    /* Raw FP32 K and V tensors: layout [layer][head][token][dim].
     * Pointer is non-owning — valid only during the call that provides it. */
    const float *k_data;       /* float[n_layers * n_heads * n_tokens * head_dim] */
    const float *v_data;
};

/* ---- Runtime metadata (returned by IRuntimeAdapter::metadata()) --------- */
struct RuntimeMetadata {
    std::string backend_name;   /* e.g. "llama_cpp_python", "transformers" */
    std::string model_name;     /* e.g. "TinyLlama-1.1B", "Qwen2-0.5B" */
    int         n_layers    = 0;
    int         n_heads     = 0;
    int         head_dim    = 0;
    int         vocab_size  = 0;
    bool        kv_access   = false; /* True if adapter can intercept KV directly */
    bool        gpu_enabled = false;

    /* AdapTQ config in use */
    int         adaptq_bits     = 4;
    int         adaptq_capacity = 4096;
};

/* ---- Generation result -------------------------------------------------- */
struct GenerationResult {
    std::vector<int32_t>    token_ids;
    std::string             text;
    int                     n_prompt_tokens    = 0;
    int                     n_generated_tokens = 0;
    double                  wall_time_ms       = 0.0;
    double                  tokens_per_sec     = 0.0;

    /* AdapTQ KV stats (populated if kv_access = true) */
    size_t  kv_bytes_adaptq  = 0;  /* Compressed KV bytes */
    size_t  kv_bytes_fp16    = 0;  /* FP16 equivalent bytes */
    double  compression_ratio = 0.0;
};

} /* namespace adaptq */
