#!/bin/bash
set -e

echo "=== RUNNING GPU VALIDATION ==="
# Qwen2-0.5B
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m adaptq.research.validation --model Qwen/Qwen2-0.5B --lengths 128 256 512 1024 2048 4096 --bits 2 3 4 --seeds 42 1337 2026 > qwen_gpu_full.log 2>&1

# TinyLlama-1.1B
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m adaptq.research.validation --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lengths 128 256 512 1024 2048 4096 --bits 2 3 4 --seeds 42 1337 2026 > tinyllama_gpu_full.log 2>&1

echo "GPU validation complete"
