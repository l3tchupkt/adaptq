"""
adaptq/research/validation.py
=============================
Comprehensive end-to-end validation for AdapTQ.

Measures:
- Perplexity (NLL) over various context lengths.
- KV Cache Memory (payload, scale, metadata, padding waste).
- Compression ratio (vs FP16).
- Prefill & Decode Tokens/sec.
- KV MSE & Cosine Similarity.
- Logit Difference (MSE vs Dense).

Usage:
  python -m adaptq.research.validation --model Qwen/Qwen2-0.5B --lengths 128 256 512 --seeds 42 123
"""
import argparse
import time
import math
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptq.runtime_py.backends.transformers_hf import AdapTQCache

def get_text_dataset(tokenizer, min_tokens=8192) -> List[int]:
    """Get a sufficiently long tokenized text for context length testing."""
    # A public domain text snippet repeated to achieve required length
    base_text = (
        "The Project Gutenberg EBook of Alice's Adventures in Wonderland, by Lewis Carroll. "
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of "
        "having nothing to do: once or twice she had peeped into the book her sister was reading, "
        "but it had no pictures or conversations in it, 'and what is the use of a book,' thought "
        "Alice 'without pictures or conversations?' So she was considering in her own mind (as well "
        "as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure "
        "of making a daisy-chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so VERY "
        "remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit "
        "say to itself, 'Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, "
        "it occurred to her that she ought to have wondered at this, but at the time it all seemed "
        "quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-POCKET, and "
        "looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind "
        "that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out "
        "of it, and burning with curiosity, she ran across the field after it, and fortunately was just "
        "in time to see it pop down a large rabbit-hole under the hedge. "
    )
    # Multiply to ensure enough tokens
    text = base_text * 100
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) < min_tokens:
        tokens = (tokens * (min_tokens // len(tokens) + 1))
    return tokens[:min_tokens]

def compute_metrics(dense_tensor: np.ndarray, approx_tensor: np.ndarray) -> Tuple[float, float, float]:
    """Compute MSE, Cosine Similarity, and Max Error."""
    dense_flat = dense_tensor.flatten()
    approx_flat = approx_tensor.flatten()
    
    mse = float(np.mean((dense_flat - approx_flat)**2))
    max_err = float(np.max(np.abs(dense_flat - approx_flat)))
    
    norm_d = np.linalg.norm(dense_flat)
    norm_a = np.linalg.norm(approx_flat)
    if norm_d > 1e-12 and norm_a > 1e-12:
        cos = float(np.dot(dense_flat, approx_flat) / (norm_d * norm_a))
    else:
        cos = 0.0
    return mse, cos, max_err

def run_validation(model_id: str, lengths: List[int], bits_list: List[int], seeds: List[int]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).eval()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", getattr(model.config, "num_kv_heads", n_heads))
    hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    head_dim = hidden_size // n_heads

    max_req_len = max(lengths) + 16 # for decoding
    print("Tokenizing dataset...", flush=True)
    t0_tok = time.perf_counter()
    full_tokens = get_text_dataset(tokenizer, max_req_len)
    print(f"Tokenization took {time.perf_counter() - t0_tok:.2f}s", flush=True)
    
    results_db = []
    
    for length in lengths:
        for seed in seeds:
            print(f"\n{'='*50}\nTesting Length: {length} | Seed: {seed}\n{'='*50}")
            
            # Set deterministic seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Sample random chunk from full tokens
            max_start = max(0, len(full_tokens) - length - 128)
            start_idx = random.randint(0, max_start)
            seq = full_tokens[start_idx : start_idx + length + 128]

            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            
            # Split into prefill (length) and decode evaluation window (128 tokens)
            prefill_ids = input_ids[:, :-128]
            decode_ids = input_ids[:, -128:]
        
        # --- 1. Run Dense Baseline ---
        print("[Dense FP16 Baseline]")
        dense_cache = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, dense_mode=True, use_pt_cache=(device=="cuda"), device=device)
        
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            prefill_out = model(prefill_ids, past_key_values=dense_cache, use_cache=True)
        torch.cuda.synchronize() if device == "cuda" else None
        prefill_time_dense = time.perf_counter() - t0
        
        # Decode step (this strictly uses the KV cache)
        with torch.no_grad():
            dense_out = model(decode_ids, past_key_values=dense_cache, use_cache=True, labels=decode_ids)
        
        dense_loss = dense_out.loss.item()
        dense_ppl = math.exp(dense_loss) if dense_loss < 100 else float('inf')
        dense_logits = dense_out.logits.cpu().numpy()

        
        # Decode 10 tokens
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(input_ids, past_key_values=dense_cache, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize() if device == "cuda" else None
        decode_time_dense = time.perf_counter() - t0
        
        # Dense Memory (Theoretical FP16)
        dense_kv_bytes = length * n_layers * n_kv_heads * head_dim * 2 * 2 # 2 (K,V) * 2 bytes/elem (FP16)
        dense_mb = dense_kv_bytes / (1024**2)
        
        results_db.append({
            "model": model_id,
            "mode": "Dense",
            "context_length": length,
            "seed": seed,
            "bits": 16,
            "nll": dense_loss,
            "perplexity": dense_ppl,
            "prefill_tok_s": length / prefill_time_dense,
            "decode_tok_s": 10 / decode_time_dense,
            "memory_mb": dense_mb,
            "compression_ratio": 1.0,
            "kv_mse": 0.0,
            "kv_cosine": 1.0,
            "logit_mse": 0.0
        })
        
        print(f"  PPL: {dense_ppl:.4f} | Mem: {dense_mb:.2f} MB | Prefill: {length/prefill_time_dense:.1f} t/s")
        
        # --- 2. Run AdapTQ for each bit width ---
        for bits in bits_list:
            print(f"\n[AdapTQ {bits}-bit]")
            # Enable dense_ref to capture ground truth for MSE calculation
            cache = AdapTQCache(n_layers, n_heads, head_dim, n_kv_heads, adaptq_bits=bits, dense_mode=False, use_pt_cache=(device=="cuda"), device=device)
            cache._packed_cache.dense_ref = True # Enable reference capturing
            cache._packed_cache._dense_k = {l: [] for l in range(n_layers)}
            cache._packed_cache._dense_v = {l: [] for l in range(n_layers)}
            
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                prefill_out = model(prefill_ids, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize() if device == "cuda" else None
            prefill_time = time.perf_counter() - t0
            
            # Decode step to calculate PPL based ONLY on the compressed cache
            with torch.no_grad():
                out = model(decode_ids, past_key_values=cache, use_cache=True, labels=decode_ids)
            
            loss = out.loss.item()
            ppl = math.exp(loss) if loss < 100 else float('inf')
            logits = out.logits.cpu().numpy()
            
            # Logit MSE
            logit_mse = float(np.mean((dense_logits - logits)**2))
            
            # KV MSE / Cosine
            total_mse, total_cos, max_err = 0.0, 0.0, 0.0
            
            # We must pull from _packed_cache.get() vs _packed_cache.get_dense_ref()
            for l in range(n_layers):
                aq_k, aq_v = cache._packed_cache.get(l)
                ref_k, ref_v = cache._packed_cache.get_dense_ref(l)
                
                k_mse, k_cos, k_max = compute_metrics(ref_k, aq_k)
                v_mse, v_cos, v_max = compute_metrics(ref_v, aq_v)
                
                total_mse += (k_mse + v_mse) / 2
                total_cos += (k_cos + v_cos) / 2
                max_err = max(max_err, k_max, v_max)
                
            avg_kv_mse = total_mse / n_layers
            avg_kv_cos = total_cos / n_layers
            
            # Decode
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(input_ids, past_key_values=cache, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize() if device == "cuda" else None
            decode_time = time.perf_counter() - t0
            
            # Memory Accounting
            stats = cache._packed_cache.memory_bytes()
            adaptq_mb = stats["total_bytes"] / (1024**2)
            ratio = stats["fp16_equivalent_bytes"] / stats["total_bytes"] if stats["total_bytes"]>0 else 0
            
            print(f"  PPL: {ppl:.4f} (Diff: {ppl-dense_ppl:+.4f}) | Mem: {adaptq_mb:.2f} MB ({ratio:.2f}x) | Logit MSE: {logit_mse:.4e}")
            print(f"  KV Metrics -> MSE: {avg_kv_mse:.4e} | Cosine: {avg_kv_cos:.4f} | MaxErr: {max_err:.2f}")
            print(f"  Speed -> Prefill: {length/prefill_time:.1f} t/s | Decode: {10/decode_time:.1f} t/s")
            
            results_db.append({
                "model": model_id,
                "mode": f"AdapTQ",
                "bits": bits,
                "context_length": length,
                "seed": seed,
                "nll": loss,
                "perplexity": ppl,
                "prefill_tok_s": length / prefill_time,
                "decode_tok_s": 10 / decode_time,
                "memory_mb": adaptq_mb,
                "compression_ratio": ratio,
                "kv_mse": avg_kv_mse,
                "kv_cosine": avg_kv_cos,
                "max_err": max_err,
                "logit_mse": logit_mse
            })
            
            # Clean up
            cache = None
            out = None
            torch.cuda.empty_cache() if device == "cuda" else None

    # Save results
    os.makedirs("adaptq/research/results", exist_ok=True)
    safe_name = model_id.replace("/", "_")
    out_path = f"adaptq/research/results/validation_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(results_db, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026])
    args = parser.parse_args()
    
    run_validation(args.model, args.lengths, args.bits, args.seeds)
