"""
PagedAttention Block Table and Virtual KV Cache Management Adapter for AdapTQ.

Provides virtual memory paging, physical block allocation, logical-to-physical
slot mapping, prefix sharing with Copy-On-Write (CoW), and block table management
modeled after the vLLM PagedAttention runtime specification.
"""

from typing import List, Dict, Optional, Tuple, Set, Any
import math
import numpy as np


class PhysicalBlock:
    """
    Represents a fixed-size physical memory block holding K and V cache vectors.
    """
    def __init__(self, block_id: int, block_size: int, head_dim: int, num_heads: int):
        self.block_id = block_id
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.ref_count = 0
        self.num_tokens = 0

        # Physical storage buffers (block_size, num_heads, head_dim)
        self.k_cache = np.zeros((block_size, num_heads, head_dim), dtype=np.float32)
        self.v_cache = np.zeros((block_size, num_heads, head_dim), dtype=np.float32)

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    def append_token(self, k: np.ndarray, v: np.ndarray) -> int:
        if self.is_full:
            raise RuntimeError(f"PhysicalBlock {self.block_id} is full ({self.block_size} tokens)")
        idx = self.num_tokens
        self.k_cache[idx] = k
        self.v_cache[idx] = v
        self.num_tokens += 1
        return idx

    def clone(self, new_block_id: int) -> "PhysicalBlock":
        new_block = PhysicalBlock(new_block_id, self.block_size, self.head_dim, self.num_heads)
        new_block.num_tokens = self.num_tokens
        new_block.k_cache[:] = self.k_cache
        new_block.v_cache[:] = self.v_cache
        return new_block

    def reset(self) -> None:
        self.ref_count = 0
        self.num_tokens = 0
        self.k_cache.fill(0.0)
        self.v_cache.fill(0.0)


class BlockAllocator:
    """
    Manages a pool of reusable physical blocks with reference-counted sharing and CoW.
    """
    def __init__(self, num_blocks: int, block_size: int, head_dim: int, num_heads: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be strictly positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(i, block_size, head_dim, num_heads) for i in range(num_blocks)
        ]
        self.free_block_ids: List[int] = list(range(num_blocks))
        self.allocated_block_ids: Set[int] = set()
        self.total_cow_copies = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self.allocated_block_ids)

    @property
    def memory_utilization(self) -> float:
        return float(self.num_allocated_blocks) / float(self.num_blocks)

    def allocate(self) -> PhysicalBlock:
        if not self.free_block_ids:
            raise MemoryError("Out of physical KV cache blocks in BlockAllocator pool")
        bid = self.free_block_ids.pop(0)
        block = self.blocks[bid]
        block.reset()
        block.ref_count = 1
        self.allocated_block_ids.add(bid)
        return block

    def fork(self, block_id: int) -> PhysicalBlock:
        if block_id not in self.allocated_block_ids:
            raise ValueError(f"Cannot fork unallocated block {block_id}")
        block = self.blocks[block_id]
        block.ref_count += 1
        return block

    def copy_on_write(self, block_id: int) -> PhysicalBlock:
        """
        Duplicate block if shared (ref_count > 1), returning private unshared block.
        """
        block = self.blocks[block_id]
        if block.ref_count <= 1:
            return block

        # Shared block: allocate new block and clone
        new_block = self.allocate()
        new_block.num_tokens = block.num_tokens
        new_block.k_cache[:] = block.k_cache
        new_block.v_cache[:] = block.v_cache

        # Decrement old block ref count
        block.ref_count -= 1
        self.total_cow_copies += 1
        return new_block

    def free(self, block_id: int) -> None:
        if block_id not in self.allocated_block_ids:
            raise ValueError(f"Double free or invalid free of block {block_id}")
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.reset()
            self.allocated_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)


class SequenceBlockTable:
    """
    Virtual block table mapping a sequence's logical token stream to physical blocks.
    """
    def __init__(self, seq_id: str, allocator: BlockAllocator):
        self.seq_id = seq_id
        self.allocator = allocator
        self.block_ids: List[int] = []
        self.total_tokens = 0

    def append_token(self, k: np.ndarray, v: np.ndarray) -> Tuple[int, int]:
        """
        Append a single token (k, v) to the sequence.
        Allocates new block if needed, or triggers Copy-on-Write if current block is shared.
        Returns (physical_block_id, slot_offset_in_block).
        """
        if not self.block_ids or self.allocator.blocks[self.block_ids[-1]].is_full:
            # Need a new physical block
            new_block = self.allocator.allocate()
            self.block_ids.append(new_block.block_id)

        last_bid = self.block_ids[-1]
        last_block = self.allocator.blocks[last_bid]

        # Check Copy-on-Write
        if last_block.ref_count > 1:
            private_block = self.allocator.copy_on_write(last_bid)
            self.block_ids[-1] = private_block.block_id
            last_block = private_block

        offset = last_block.append_token(k, v)
        self.total_tokens += 1
        return last_block.block_id, offset

    def get_token(self, logical_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        if logical_pos < 0 or logical_pos >= self.total_tokens:
            raise IndexError(f"Logical position {logical_pos} out of range [0, {self.total_tokens})")
        block_idx = logical_pos // self.allocator.block_size
        offset = logical_pos % self.allocator.block_size
        bid = self.block_ids[block_idx]
        block = self.allocator.blocks[bid]
        return block.k_cache[offset], block.v_cache[offset]

    def to_block_table(self) -> List[int]:
        """Return list of physical block numbers for vLLM PagedAttention kernel."""
        return list(self.block_ids)

    def free(self) -> None:
        """Release all mapped blocks back to the allocator."""
        for bid in self.block_ids:
            self.allocator.free(bid)
        self.block_ids.clear()
        self.total_tokens = 0


class PagedKVCacheManager:
    """
    High-level orchestrator managing multi-sequence PagedAttention KV caches,
    prefix caching, and memory defragmentation.
    """
    def __init__(self, num_blocks: int = 128, block_size: int = 16, head_dim: int = 64, num_heads: int = 8):
        self.allocator = BlockAllocator(num_blocks, block_size, head_dim, num_heads)
        self.sequences: Dict[str, SequenceBlockTable] = {}
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_heads = num_heads

    def create_sequence(self, seq_id: str) -> SequenceBlockTable:
        if seq_id in self.sequences:
            raise ValueError(f"Sequence {seq_id} already exists")
        table = SequenceBlockTable(seq_id, self.allocator)
        self.sequences[seq_id] = table
        return table

    def append_token(self, seq_id: str, k: np.ndarray, v: np.ndarray) -> Tuple[int, int]:
        if seq_id not in self.sequences:
            raise KeyError(f"Sequence {seq_id} not found")
        return self.sequences[seq_id].append_token(k, v)

    def fork_sequence(self, source_seq_id: str, new_seq_id: str) -> SequenceBlockTable:
        """
        Create a new sequence sharing all fully-filled blocks with the source sequence
        using zero-copy reference counting (Prefix Caching).
        """
        if source_seq_id not in self.sequences:
            raise KeyError(f"Source sequence {source_seq_id} does not exist")
        if new_seq_id in self.sequences:
            raise ValueError(f"Destination sequence {new_seq_id} already exists")

        src_table = self.sequences[source_seq_id]
        forked_table = SequenceBlockTable(new_seq_id, self.allocator)

        for bid in src_table.block_ids:
            self.allocator.fork(bid)
            forked_table.block_ids.append(bid)

        forked_table.total_tokens = src_table.total_tokens
        self.sequences[new_seq_id] = forked_table
        return forked_table

    def free_sequence(self, seq_id: str) -> None:
        if seq_id in self.sequences:
            self.sequences[seq_id].free()
            del self.sequences[seq_id]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_blocks": self.allocator.num_blocks,
            "allocated_blocks": self.allocator.num_allocated_blocks,
            "free_blocks": self.allocator.num_free_blocks,
            "utilization_pct": round(self.allocator.memory_utilization * 100.0, 2),
            "total_sequences": len(self.sequences),
            "total_cow_copies": self.allocator.total_cow_copies,
        }
