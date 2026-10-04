#!/bin/bash
export HF_HUB_DISABLE_PROGRESS_BARS=1
python3 -m adaptq.research.validation --model Qwen/Qwen2-0.5B --lengths 128 256 512 1024 2048 4096 > qwen_val.log 2>&1
echo "QWEN DONE: $?"
python3 -m adaptq.research.validation --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lengths 128 256 512 1024 2048 4096 > tinyllama_val.log 2>&1
echo "TINYLLAMA DONE: $?"
