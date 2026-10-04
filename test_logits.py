import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptq.runtime_py.backends.transformers_hf import AdapTQCache

model_id = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32).eval()

prompt = "The capital of France is Paris, but the capital of Italy is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
prefill_ids = input_ids[:, :-1]
decode_id = input_ids[:, -1:]

n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
head_dim = model.config.hidden_size // n_heads

cache_dense = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, dense_mode=True)
with torch.no_grad():
    model(prefill_ids, past_key_values=cache_dense, use_cache=True)
    dense_out = model(decode_id, past_key_values=cache_dense, use_cache=True)

cache_2bit = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, adaptq_bits=2, dense_mode=False)
with torch.no_grad():
    model(prefill_ids, past_key_values=cache_2bit, use_cache=True)
    out_2bit = model(decode_id, past_key_values=cache_2bit, use_cache=True)

dense_logits = dense_out.logits.numpy()
logits_2bit = out_2bit.logits.numpy()

logit_mse = float(np.mean((dense_logits - logits_2bit)**2))
print("Logit MSE:", logit_mse)
print("Are logits EXACTLY identical?", np.array_equal(dense_logits, logits_2bit))
