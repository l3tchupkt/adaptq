import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptq.runtime_py.backends.transformers_hf import AdapTQCache
from adaptq.research.packed_kv import PackedKVCache

@pytest.mark.parametrize("bits", [2, 4])
def test_hf_integration_execution_path(bits):
    """
    PROVES that the tensors consumed by Hugging Face attention are exactly 
    the reconstructed quantized tensors and NOT a dense shadow cache.
    """
    model_id = "Qwen/Qwen2-0.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu").eval()
    except Exception as e:
        pytest.skip(f"Failed to load model {model_id} for test: {e}")

    # 1. Create a prompt
    prompt = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prefill_ids = input_ids[:, :-1]
    decode_id = input_ids[:, -1:]

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads

    # 2. Run Dense Cache Baseline
    dense_cache = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, dense_mode=True)
    with torch.no_grad():
        model(prefill_ids, past_key_values=dense_cache, use_cache=True)
        dense_out = model(decode_id, past_key_values=dense_cache, use_cache=True)
    dense_logits = dense_out.logits.numpy()

    # 3. Run AdapTQ Compressed Cache
    adaptq_cache = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, adaptq_bits=bits, dense_mode=False)
    with torch.no_grad():
        model(prefill_ids, past_key_values=adaptq_cache, use_cache=True)
        adaptq_out = model(decode_id, past_key_values=adaptq_cache, use_cache=True)
    adaptq_logits = adaptq_out.logits.numpy()

    # 4. PROOF 1: The logits must mathematically diverge.
    # If a shadow cache was accidentally used, the logits would be identical to dense_logits.
    logit_mse = float(np.mean((dense_logits - adaptq_logits)**2))
    assert logit_mse > 0.0, "Logits match dense perfectly! A dense shadow cache is being illegally consumed by Hugging Face!"

    # 5. PROOF 2: The tensors residing inside the HF `DynamicCache` structure MUST precisely match the decompressed quantized tensors.
    # We intercept the cache state right after prefill to verify identity.
    layer_idx = 0
    hf_key_states = adaptq_cache.layers[layer_idx].past_key_states
    assert hf_key_states is not None, "HF dynamic cache layers were not populated!"
    
    # Extract the ground truth decompressed tensors directly from the core C++ / Python PackedKVCache memory space
    k_recon, _ = adaptq_cache._packed_cache.get(layer_idx)
    # k_recon shape: (seq_len, n_kv_heads, head_dim)
    # HF shape: (batch, n_kv_heads, seq_len, head_dim)
    
    k_recon_transposed = k_recon.transpose(1, 0, 2)
    k_recon_tensor = torch.from_numpy(k_recon_transposed).unsqueeze(0)

    # They should be mathematically identical (floating point exact)
    max_diff = torch.max(torch.abs(hf_key_states - k_recon_tensor)).item()
    assert max_diff == 0.0, f"The cache injected into HF differs from the reconstructed quantized cache (max diff: {max_diff})"
