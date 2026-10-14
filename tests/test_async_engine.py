import asyncio
import unittest
import numpy as np

from adaptq.async_engine import AsyncAdapTQEngine, AsyncEngineConfig


class TestAsyncAdapTQEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = AsyncEngineConfig(
            dim=16,
            bits=4,
            capacity=100,
            max_workers=2,
            concurrency_limit=4
        )
        self.engine = AsyncAdapTQEngine(self.config)

    async def asyncTearDown(self):
        if not self.engine.is_closed:
            await self.engine.close()

    async def test_config_validation(self):
        with self.assertRaises(ValueError):
            AsyncAdapTQEngine(AsyncEngineConfig(dim=0))
        with self.assertRaises(ValueError):
            AsyncAdapTQEngine(AsyncEngineConfig(bits=5))
        with self.assertRaises(ValueError):
            AsyncAdapTQEngine(AsyncEngineConfig(capacity=-1))
        with self.assertRaises(ValueError):
            AsyncAdapTQEngine(AsyncEngineConfig(max_workers=0))

    async def test_async_context_manager(self):
        async with AsyncAdapTQEngine(self.config) as eng:
            self.assertFalse(eng.is_closed)
        self.assertTrue(eng.is_closed)

    async def test_append_and_compute_single(self):
        k = [0.1] * 16
        v = [0.5] * 16
        pos = await self.engine.append_async(k, v, 0)
        self.assertEqual(pos, 0)

        q = [0.1] * 16
        out = await self.engine.compute_async(q)
        self.assertEqual(out.shape, (16,))
        self.assertAlmostEqual(float(out[0]), 0.5, places=3)
        self.assertEqual(self.engine.metrics.total_appends, 1)
        self.assertEqual(self.engine.metrics.total_computes, 1)

    async def test_batch_compute(self):
        for i in range(5):
            k = [float(i)] * 16
            v = [float(i + 1)] * 16
            await self.engine.append_async(k, v, i)

        queries = [[1.0] * 16, [2.0] * 16, [3.0] * 16]
        outs = await self.engine.batch_compute_async(queries)
        self.assertEqual(len(outs), 3)
        for out in outs:
            self.assertEqual(out.shape, (16,))
        self.assertEqual(self.engine.metrics.total_batch_queries, 3)

    async def test_concurrent_appends(self):
        tasks = [
            self.engine.append_async([float(i)] * 16, [float(i)] * 16, i)
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)
        self.assertEqual(len(results), 20)
        self.assertEqual(self.engine.metrics.total_appends, 20)

    async def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            await self.engine.append_async([1.0] * 8, [1.0] * 16, 0)
        with self.assertRaises(ValueError):
            await self.engine.compute_async([1.0] * 32)

    async def test_operations_on_closed_engine(self):
        await self.engine.close()
        with self.assertRaises(RuntimeError):
            await self.engine.append_async([1.0] * 16, [1.0] * 16, 0)
        with self.assertRaises(RuntimeError):
            await self.engine.compute_async([1.0] * 16)


if __name__ == "__main__":
    unittest.main()
