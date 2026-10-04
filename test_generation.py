import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptq.runtime_py.backends.transformers_hf import AdapTQCache

model_id = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32).eval()

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
head_dim = model.config.hidden_size // n_heads

cache_dense = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, dense_mode=True)
out_dense = model.generate(input_ids, past_key_values=cache_dense, max_new_tokens=10, do_sample=False)
text_dense = tokenizer.decode(out_dense[0])
print("Dense generation:", text_dense)

cache_2bit = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, adaptq_bits=2, dense_mode=False)
out_2bit = model.generate(input_ids, past_key_values=cache_2bit, max_new_tokens=10, do_sample=False)
text_2bit = tokenizer.decode(out_2bit[0])
print("2-bit generation:", text_2bit)

print("Are outputs identical?", text_dense == text_2bit)
if len(cache_dense.layers) > 0 and len(cache_2bit.layers) > 0:
    print("Layer 0 Key MSE:", torch.mean((cache_dense.layers[0].past_key_states - cache_2bit.layers[0].past_key_states)**2).item())

