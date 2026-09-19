import unittest
import numpy as np
from adaptq.paged_cache import (
    PhysicalBlock,
    BlockAllocator,
    SequenceBlockTable,
    PagedKVCacheManager,
)


class TestPagedCache(unittest.TestCase):
    def setUp(self):
        self.num_heads = 4
        self.head_dim = 16
        self.block_size = 4
        self.num_blocks = 8

    def test_physical_block_lifecycle(self):
        block = PhysicalBlock(0, self.block_size, self.head_dim, self.num_heads)
        self.assertFalse(block.is_full)
        self.assertEqual(block.num_tokens, 0)

        # Append tokens up to capacity
        for i in range(self.block_size):
            k = np.ones((self.num_heads, self.head_dim), dtype=np.float32) * (i + 1)
            v = np.ones((self.num_heads, self.head_dim), dtype=np.float32) * (i + 10)
            idx = block.append_token(k, v)
            self.assertEqual(idx, i)

        self.assertTrue(block.is_full)
        self.assertEqual(block.num_tokens, self.block_size)

        # Appending when full must raise RuntimeError
        k_extra = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        v_extra = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            block.append_token(k_extra, v_extra)

        # Clone block
        clone = block.clone(new_block_id=1)
        self.assertEqual(clone.block_id, 1)
        self.assertEqual(clone.num_tokens, self.block_size)
        np.testing.assert_array_equal(clone.k_cache, block.k_cache)

        # Reset block
        block.reset()
        self.assertEqual(block.num_tokens, 0)
        self.assertFalse(block.is_full)
        self.assertTrue(np.all(block.k_cache == 0.0))

    def test_block_allocator_pool_and_exhaustion(self):
        allocator = BlockAllocator(self.num_blocks, self.block_size, self.head_dim, self.num_heads)
        self.assertEqual(allocator.num_free_blocks, self.num_blocks)
        self.assertEqual(allocator.num_allocated_blocks, 0)
        self.assertAlmostEqual(allocator.memory_utilization, 0.0)

        allocated = []
        for _ in range(self.num_blocks):
            allocated.append(allocator.allocate())

        self.assertEqual(allocator.num_free_blocks, 0)
        self.assertEqual(allocator.num_allocated_blocks, self.num_blocks)
        self.assertAlmostEqual(allocator.memory_utilization, 1.0)

        # Allocating when pool is empty must raise MemoryError
        with self.assertRaises(MemoryError):
            allocator.allocate()

        # Free all blocks
        for b in allocated:
            allocator.free(b.block_id)

        self.assertEqual(allocator.num_free_blocks, self.num_blocks)
        self.assertEqual(allocator.num_allocated_blocks, 0)

    def test_sequence_block_table_token_append_and_mapping(self):
        allocator = BlockAllocator(self.num_blocks, self.block_size, self.head_dim, self.num_heads)
        table = SequenceBlockTable("seq_1", allocator)

        total_tokens_to_add = 10  # Needs ceil(10/4) = 3 blocks
        for i in range(total_tokens_to_add):
            k = np.full((self.num_heads, self.head_dim), float(i), dtype=np.float32)
            v = np.full((self.num_heads, self.head_dim), float(i * 2), dtype=np.float32)
            table.append_token(k, v)

        self.assertEqual(table.total_tokens, total_tokens_to_add)
        self.assertEqual(len(table.block_ids), 3)

        # Check retrieval at specific logical positions
        for i in range(total_tokens_to_add):
            k_ret, v_ret = table.get_token(i)
            self.assertTrue(np.all(k_ret == float(i)))
            self.assertTrue(np.all(v_ret == float(i * 2)))

        # Out of bounds access raises IndexError
        with self.assertRaises(IndexError):
            table.get_token(10)

        table.free()
        self.assertEqual(table.total_tokens, 0)
        self.assertEqual(len(table.block_ids), 0)
        self.assertEqual(allocator.num_free_blocks, self.num_blocks)

    def test_copy_on_write_prefix_sharing(self):
        manager = PagedKVCacheManager(
            num_blocks=16,
            block_size=self.block_size,
            head_dim=self.head_dim,
            num_heads=self.num_heads
        )

        seq1 = manager.create_sequence("seq_base")

        # Ingest 4 prompt tokens (fills exactly 1 block)
        for i in range(4):
            k = np.full((self.num_heads, self.head_dim), float(i), dtype=np.float32)
            v = np.full((self.num_heads, self.head_dim), float(i + 10), dtype=np.float32)
            manager.append_token("seq_base", k, v)

        self.assertEqual(len(seq1.block_ids), 1)
        shared_bid = seq1.block_ids[0]
        self.assertEqual(manager.allocator.blocks[shared_bid].ref_count, 1)

        # Fork sequence 2 from sequence 1 (prefix caching)
        seq2 = manager.fork_sequence("seq_base", "seq_forked")
        self.assertEqual(manager.allocator.blocks[shared_bid].ref_count, 2)
        self.assertEqual(seq1.block_ids, seq2.block_ids)

        # Now append token to seq2: should trigger Copy-on-Write
        k_branch = np.full((self.num_heads, self.head_dim), 99.0, dtype=np.float32)
        v_branch = np.full((self.num_heads, self.head_dim), 999.0, dtype=np.float32)
        manager.append_token("seq_forked", k_branch, v_branch)

        # Seq2 should now have allocated a new private block
        self.assertEqual(len(seq2.block_ids), 2)
        self.assertNotEqual(seq2.block_ids[1], shared_bid)

        # Verify stats
        stats = manager.get_stats()
        self.assertEqual(stats["total_sequences"], 2)
        self.assertGreater(stats["total_blocks"], 0)

        # Free sequences
        manager.free_sequence("seq_base")
        manager.free_sequence("seq_forked")
        self.assertEqual(manager.allocator.num_free_blocks, 16)


if __name__ == "__main__":
    unittest.main()
