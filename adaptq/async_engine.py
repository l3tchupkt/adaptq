"""
adaptq.async_engine — Asynchronous Python Inference Worker & Batch Pipeline

Provides an asyncio-compatible engine for high-throughput non-blocking
KV cache operations, concurrency-bounded worker dispatch, and batch evaluation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class AsyncEngineConfig:
    dim: int = 64
    bits: int = 4
    capacity: int = 4096
    max_workers: int = 4
    concurrency_limit: int = 16
    queue_timeout_seconds: float = 30.0


@dataclass
class EngineMetrics:
    total_appends: int = 0
    total_computes: int = 0
    total_batch_queries: int = 0
    total_errors: int = 0
    total_wait_time_seconds: float = 0.0


class AsyncAdapTQEngine:
    """Asynchronous pipeline engine wrapping AdapTQ quantized KV cache."""

    def __init__(self, config: Optional[AsyncEngineConfig] = None) -> None:
        self.config = config or AsyncEngineConfig()
        if self.config.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.config.dim}")
        if self.config.bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {self.config.bits}")
        if self.config.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.config.capacity}")
        if self.config.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.config.max_workers}")

        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="adaptq-async-worker"
        )
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._is_closed = False
        self._metrics = EngineMetrics()
        self._cache_storage: List[Tuple[np.ndarray, np.ndarray]] = []
        self._lock = asyncio.Lock()

    async def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.concurrency_limit)
        return self._semaphore

    async def __aenter__(self) -> "AsyncAdapTQEngine":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def append_async(
        self,
        key: Sequence[float],
        val: Sequence[float],
        token_pos: int
    ) -> int:
        """Asynchronously append a key-value vector pair to the quantized cache."""
        if self._is_closed:
            raise RuntimeError("Cannot append: AsyncAdapTQEngine is closed")

        k_arr = np.asarray(key, dtype=np.float32)
        v_arr = np.asarray(val, dtype=np.float32)

        if k_arr.size != self.config.dim or v_arr.size != self.config.dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.config.dim}, "
                f"got key={k_arr.size}, val={v_arr.size}"
            )
        if token_pos < 0:
            raise ValueError(f"token_pos must be >= 0, got {token_pos}")

        sem = await self._get_semaphore()
        t0 = time.perf_counter()

        async with sem:
            wait_duration = time.perf_counter() - t0
            self._metrics.total_wait_time_seconds += wait_duration

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                self._sync_append_worker,
                k_arr,
                v_arr,
                token_pos
            )

        async with self._lock:
            self._metrics.total_appends += 1

        return token_pos

    def _sync_append_worker(self, k: np.ndarray, v: np.ndarray, pos: int) -> None:
        # Worker thread simulation/quantization step
        if len(self._cache_storage) >= self.config.capacity:
            self._cache_storage.pop(0) # FIFO eviction
        self._cache_storage.append((k, v))

    async def compute_async(self, query: Sequence[float]) -> np.ndarray:
        """Asynchronously compute attention output vector against cached KV pairs."""
        if self._is_closed:
            raise RuntimeError("Cannot compute: AsyncAdapTQEngine is closed")

        q_arr = np.asarray(query, dtype=np.float32)
        if q_arr.size != self.config.dim:
            raise ValueError(f"Query dim mismatch: expected {self.config.dim}, got {q_arr.size}")

        sem = await self._get_semaphore()
        async with sem:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(
                self._executor,
                self._sync_compute_worker,
                q_arr
            )

        async with self._lock:
            self._metrics.total_computes += 1

        return out

    def _sync_compute_worker(self, q: np.ndarray) -> np.ndarray:
        if not self._cache_storage:
            return np.zeros(self.config.dim, dtype=np.float32)

        # Vectorized dot product & weighted sum simulation
        k_stacked = np.stack([k for k, _ in self._cache_storage])
        v_stacked = np.stack([v for _, v in self._cache_storage])

        scores = np.matmul(k_stacked, q) / np.sqrt(self.config.dim)
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / (np.sum(exp_scores) + 1e-12)

        out = np.sum(v_stacked * weights[:, np.newaxis], axis=0).astype(np.float32)
        return out

    async def batch_compute_async(
        self,
        queries: Sequence[Sequence[float]]
    ) -> List[np.ndarray]:
        """Compute multiple attention queries concurrently across workers."""
        if self._is_closed:
            raise RuntimeError("Cannot batch compute: engine is closed")
        if not queries:
            return []

        tasks = [self.compute_async(q) for q in queries]
        results = await asyncio.gather(*tasks)

        async with self._lock:
            self._metrics.total_batch_queries += len(queries)

        return list(results)

    async def close(self) -> None:
        """Shutdown thread pool and release engine resources."""
        if not self._is_closed:
            self._is_closed = True
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._executor.shutdown, True)
            self._cache_storage.clear()

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics

    @property
    def is_closed(self) -> bool:
        return self._is_closed
