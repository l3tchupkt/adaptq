#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "runtime_metadata.h"

/* =========================================================================
 * runtime/adapters/adapter.h — IRuntimeAdapter
 *
 * Backend-agnostic adapter interface for integrating AdapTQ into any
 * LLM inference engine.
 *
 * DESIGN PRINCIPLES:
 *   - AdapTQ core (RuntimeContext, SessionSnapshot, ReplayEngine) is NEVER
 *     modified to accommodate a backend. Only adapters change.
 *   - All methods operate on plain types: float*, int32_t*, std::string.
 *     No engine-specific tensor types cross this boundary.
 *   - Adapters are NOT thread-safe. Use one adapter instance per thread.
 *
 * IMPLEMENTATION GUIDE:
 *   To add a new backend, subclass IRuntimeAdapter and implement the 9 pure
 *   virtual methods below. See docs/v2.1_integration.md for a full walkthrough.
 *
 * USAGE:
 *   auto adapter = create_adapter("llama_cpp_python");
 *   adapter->load_model(model_cfg);
 *   adapter->begin_session(session_cfg);
 *   adapter->prefill(prompt_tokens);
 *   int32_t tok;
 *   while (adapter->decode_next(&tok) && tok != eos) output.push_back(tok);
 *   auto result = adapter->generation_result();
 *   adapter->end_session();
 *
 * ======================================================================= */

namespace adaptq {

class SessionSnapshot;   /* forward */
class RuntimeContext;    /* forward */

class IRuntimeAdapter {
public:
    virtual ~IRuntimeAdapter() = default;

    /* ------------------------------------------------------------------ */
    /* Model lifecycle                                                      */
    /* ------------------------------------------------------------------ */

    /**
     * Load model weights.
     * @return true on success.
     */
    virtual bool load_model(const ModelConfig &cfg) = 0;

    /**
     * Load or initialize tokenizer.
     * For most backends, tokenizer_path == model_path (no-op to call separately).
     * @return true on success.
     */
    virtual bool load_tokenizer(const std::string &tokenizer_path) = 0;

    /* ------------------------------------------------------------------ */
    /* Session lifecycle                                                    */
    /* ------------------------------------------------------------------ */

    /**
     * Begin a new inference session.
     * Resets KV cache and prepares for tokenization + generation.
     * @return true on success.
     */
    virtual bool begin_session(const SessionConfig &cfg) = 0;

    /**
     * End the current session.
     * Releases session-local resources. Model weights are NOT unloaded.
     * @return true on success.
     */
    virtual bool end_session() = 0;

    /* ------------------------------------------------------------------ */
    /* Generation pipeline                                                  */
    /* ------------------------------------------------------------------ */

    /**
     * Prefill: feed all prompt tokens into the KV cache.
     * Internally calls RuntimeContext::append() per layer per head.
     * @param tokens  Tokenized prompt.
     * @return true on success.
     */
    virtual bool prefill(const std::vector<int32_t> &tokens) = 0;

    /**
     * Decode: generate the next token.
     * Internally calls RuntimeContext::compute() and updates KV cache.
     * @param out_token  Written with the generated token ID.
     * @return true while generation should continue; false on EOS or error.
     */
    virtual bool decode_next(int32_t *out_token) = 0;

    /* ------------------------------------------------------------------ */
    /* KV cache access (for snapshot integration)                           */
    /* ------------------------------------------------------------------ */

    /**
     * Get a non-owning view of the current KV cache.
     * @param out  Filled with layer/head/dim counts and raw float pointers.
     * @return true if this backend supports direct KV access; false otherwise.
     *
     * Backends that return false (e.g. Ollama REST) capture only
     * prompt/output at the Python level.
     */
    virtual bool get_kv_cache(KVCacheView *out) = 0;

    /**
     * Restore KV cache from a view.
     * Used after loading a SessionSnapshot to resume generation.
     * @return true if this backend supports KV restore; false otherwise.
     */
    virtual bool set_kv_cache(const KVCacheView &kv) = 0;

    /**
     * Clear the KV cache (same as new session without re-loading model).
     * @return true on success.
     */
    virtual bool clear_kv_cache() = 0;

    /* ------------------------------------------------------------------ */
    /* Metadata and diagnostics                                             */
    /* ------------------------------------------------------------------ */

    /**
     * Returns backend/model metadata populated after load_model().
     */
    virtual RuntimeMetadata metadata() const = 0;

    /**
     * Returns the last error message, or empty string if no error.
     */
    virtual std::string last_error() const { return ""; }

    /**
     * Returns generation statistics for the last completed session.
     * Valid after at least one decode_next() call.
     */
    virtual GenerationResult generation_result() const = 0;
};

} /* namespace adaptq */
