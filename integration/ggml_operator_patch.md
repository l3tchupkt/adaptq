# AdapTQ Native GGML Operator Integration

## Overview
This document represents the absolute correct integration strategy into `llama.cpp`'s native architecture. Since modern `ggml` uses a static computation graph execution, directly invoking attention loops during graph construction via C++ fails. 

We must define a custom computation node: `GGML_OP_ADAPTQ_ATTN`. This operator executes purely during `ggml_graph_compute` on CPU backends by interfacing cleanly with our `adaptq_compute_batch` C hooks.

---

### Phase 1: Modifying `ggml.h`
Open `ggml/include/ggml.h`. Add the operator enumeration and function prototype.

**Location**: Inside `enum ggml_op`
```diff
     GGML_OP_CROSS_ENTROPY_LOSS,
     GGML_OP_CROSS_ENTROPY_LOSS_BACK,
     GGML_OP_OPT_STEP_ADAMW,
+    GGML_OP_ADAPTQ_ATTN,     // AdapTQ Attention Forward Pass
     
     GGML_OP_COUNT,
 };
```

**Location**: At the end of the graph builder functions
```c
// --- AdapTQ Integration ---
// Builds an attention op against an opaque adaptq_mha_t layer context.
// Passes the pointer inside the op_params struct dynamically.
GGML_API struct ggml_tensor * ggml_adaptq_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        void                * adaptq_mha);
```

---

### Phase 2: Modifying `ggml.c`
Open `ggml/src/ggml.c`. Define tensor shape inference rules and parameter parsing.

**Location**: `ggml_compute_forward` (or `ggml_compute_forward_op` name resolver)
```diff
     case GGML_OP_CROSS_ENTROPY_LOSS_BACK: return "CROSS_ENTROPY_LOSS_BACK";
     case GGML_OP_OPT_STEP_ADAMW:      return "OPT_STEP_ADAMW";
+    case GGML_OP_ADAPTQ_ATTN:         return "ADAPTQ_ATTN";
```

**Location**: Implementing `ggml_adaptq_attn`
Insert this near the other operation creators:
```c
struct ggml_tensor * ggml_adaptq_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        void                * adaptq_mha) {

    // Output is shape of Queries: [head_dim, n_heads, n_queries, 1]
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, q->ne);
    result->op = GGML_OP_ADAPTQ_ATTN;
    result->src[0] = q;

    // We inject the MHA void pointer into op_params directly.
    void* params[2] = { adaptq_mha, NULL };  // 16 bytes max (op_params is size 64)
    ggml_set_op_params(result, params, sizeof(params));

    return result;
}
```

---

### Phase 3: Modifying `ggml-cpu.c` (The Kernel Dispatch)
Open `ggml/src/ggml-cpu.c`. This is where the forward pass evaluates at runtime.

Add your header reference to the top:
```c
#include "adaptq.h"
```

**Location**: Define the compute kernel (e.g. `ggml_compute_forward_adaptq_attn`)
```c
static void ggml_compute_forward_adaptq_attn(
        struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * q = dst->src[0];

    // Restore pointer from op_params
    void * mha = NULL;
    memcpy(&mha, dst->op_params, sizeof(void*));

    if (!mha) {
        return; // Empty cache
    }

    const int n_heads   = q->ne[1];
    const int n_queries = q->ne[2];
    const int head_dim  = q->ne[0];

    // In modern ggml, threads slice parallel operations across queries or heads.
    // For simplicity, we delegate iteration entirely to AdapTQ's OpenMP core
    // and execute everything under thread 0.
    if (params->ith != 0) return;

    for (int h = 0; h < n_heads; ++h) {
        const float * q_data   = (const float *)q->data + h * head_dim;
        float * out_data       = (float *)dst->data + h * head_dim;

        adaptq_mha_compute_batch(
            (adaptq_mha_t)mha, 
            h, 
            q_data, 
            n_queries, 
            out_data);
    }
}
```

**Location**: Bind it in `ggml_compute_forward_cpu`
```diff
     case GGML_OP_OPT_STEP_ADAMW:
         ggml_compute_forward_opt_step_adamw(params, tensor);
         break;
+    case GGML_OP_ADAPTQ_ATTN:
+        ggml_compute_forward_adaptq_attn(params, tensor);
+        break;
     default:
         GGML_ABORT("fatal error");
 }
```

---

### Phase 4: Modifying `llama.cpp` (Graph Interpolation)
Open `src/llama.cpp`. Find the KV caching layer `llm_build_kv_store` and attention layer `llm_build_kqv`.

**Location**: Context handling (`struct llama_kv_cache`)
*(Same logic mapped previously: Inject `void* adaptq_mha[128]` into `kv_cache_self`, along with `adaptq_enabled` config flag parameters)*.

**Location**: `llm_build_kv_store`
Instead of copying QK to GGML tensors, intercept the output dynamically:
```cpp
    // AdapTQ KV Population (happens per generation token or prompt burst)
    if (kv_self.adaptq_enabled) {
        // Warning: This executes synchronously during graph building! 
        // This is safe ONLY IF k_cur/v_cur are immediately available 
        // OR if you defer via another GGML_OP_ADAPTQ_STORE. 
        // To properly defer: build GGML_OP_ADAPTQ_STORE!
    }
```
*Wait! Actually, if GGML builds graphs beforehand... K/V additions must also be nodes!*

We must similarly defer `append` to a `GGML_OP_ADAPTQ_APPEND`:
```c
// ggml.c
struct ggml_tensor * ggml_adaptq_append(
    struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v, void* mha) {
    struct ggml_tensor * res = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    res->op = GGML_OP_ADAPTQ_APPEND;
    res->src[0] = k; res->src[1] = v;
    void* params[2] = {mha, NULL};
    ggml_set_op_params(res, params, sizeof(params));
    return res;
}
```

**Location**: `llm_build_kqv`
Replace `ggml_flash_attn_ext` mapping entirely:
```cpp
    struct ggml_tensor * kqv_out = NULL;
    
    if (lctx.kv_self.adaptq_enabled) {
        // Emit Append Nodes first
        struct ggml_tensor * kv_op = ggml_adaptq_append(ctx, cur_k, cur_v, lctx.kv_self.adaptq_layers[il]);
        ggml_build_forward_expand(ctx, kv_op);
        
        // Emit Compute Nodes
        kqv_out = ggml_adaptq_attn(ctx, cur_q, lctx.kv_self.adaptq_layers[il]);
        
    } else {
        // Normal GGML Flash Attention
        kqv_out = ggml_flash_attn_ext(ctx, cur_q, cur_k, cur_v, ...);
    }
```

---

### Build Instructions
Clone `llama.cpp` using local integration paths:
```bash
git clone https://github.com/ggerganov/llama.cpp.git

# Copy the AdapTQ library dependencies
mkdir -p llama.cpp/adaptq
cp /path/to/adapTQ/libadaptq.so llama.cpp/adaptq/
cp /path/to/adapTQ/include/adaptq.h llama.cpp/adaptq/

# Modify CMakeLists.txt inside llama.cpp
# Add:
# target_include_directories(ggml PRIVATE ${CMAKE_SOURCE_DIR}/adaptq)
# target_link_libraries(ggml PRIVATE ${CMAKE_SOURCE_DIR}/adaptq/libadaptq.so)
# target_link_libraries(llama PRIVATE ${CMAKE_SOURCE_DIR}/adaptq/libadaptq.so)

# Build
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_AVX2=ON -DCMAKE_BUILD_TYPE=Release
make -j
```

```bash
# Execute local benchmarking
./bin/main -m tinyllama-1.1b-chat.gguf --use-adaptq --ctx-size 8192 -p "Tell me about large language models" -n 200
```
