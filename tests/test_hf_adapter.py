import unittest
import numpy as np

from adaptq.hf_adapter import AdapTQCache


class TestAdapTQCache(unittest.TestCase):
    def setUp(self):
        self.num_layers = 4
        self.head_dim = 32
        self.capacity = 20
        self.cache = AdapTQCache(
            num_hidden_layers=self.num_layers,
            bits=4,
            max_capacity=self.capacity,
            head_dim=self.head_dim
        )

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            AdapTQCache(num_hidden_layers=0)
        with self.assertRaises(ValueError):
            AdapTQCache(bits=5)
        with self.assertRaises(ValueError):
            AdapTQCache(max_capacity=-5)
        with self.assertRaises(ValueError):
            AdapTQCache(head_dim=0)

    def test_initial_state(self):
        self.assertEqual(self.cache.get_seq_length(0), 0)
        self.assertEqual(self.cache.get_seq_length(99), 0)  # Out of bounds safe
        self.assertEqual(self.cache.get_max_length(), self.capacity)
        self.assertEqual(self.cache.compression_factor(), 1.0)

    def test_single_update(self):
        # Shape: [batch=1, heads=2, seq=5, dim=32]
        k = np.ones((1, 2, 5, self.head_dim), dtype=np.float32)
        v = np.ones((1, 2, 5, self.head_dim), dtype=np.float32) * 2.0

        k_out, v_out = self.cache.update(k, v, layer_idx=0)
        self.assertEqual(k_out.shape, (1, 2, 5, self.head_dim))
        self.assertEqual(v_out.shape, (1, 2, 5, self.head_dim))
        self.assertEqual(self.cache.get_seq_length(0), 5)
        self.assertEqual(self.cache.get_seq_length(1), 0)

    def test_multi_step_concatenation(self):
        k1 = np.ones((1, 2, 3, self.head_dim), dtype=np.float32)
        v1 = np.ones((1, 2, 3, self.head_dim), dtype=np.float32)
        self.cache.update(k1, v1, layer_idx=0)

        k2 = np.ones((1, 2, 4, self.head_dim), dtype=np.float32) * 3.0
        v2 = np.ones((1, 2, 4, self.head_dim), dtype=np.float32) * 3.0
        k_out, v_out = self.cache.update(k2, v2, layer_idx=0)

        self.assertEqual(k_out.shape, (1, 2, 7, self.head_dim))
        self.assertEqual(self.cache.get_seq_length(0), 7)

    def test_capacity_truncation(self):
        # Exceed max_capacity of 20
        k = np.ones((1, 1, 25, self.head_dim), dtype=np.float32)
        v = np.ones((1, 1, 25, self.head_dim), dtype=np.float32)
        k_out, v_out = self.cache.update(k, v, layer_idx=0)

        self.assertEqual(k_out.shape[2], self.capacity)
        self.assertEqual(self.cache.get_seq_length(0), self.capacity)

    def test_crop_and_usable_length(self):
        k = np.ones((1, 1, 15, self.head_dim), dtype=np.float32)
        v = np.ones((1, 1, 15, self.head_dim), dtype=np.float32)
        self.cache.update(k, v, layer_idx=0)

        usable = self.cache.get_usable_length(new_seq_length=10, layer_idx=0)
        self.assertEqual(usable, 10)

        self.cache.crop(max_length=8)
        self.assertEqual(self.cache.get_seq_length(0), 8)

    def test_to_legacy_cache(self):
        k = np.ones((1, 1, 4, self.head_dim), dtype=np.float32)
        v = np.ones((1, 1, 4, self.head_dim), dtype=np.float32)
        self.cache.update(k, v, layer_idx=1)

        legacy = self.cache.to_legacy_cache()
        self.assertEqual(len(legacy), self.num_layers)
        self.assertEqual(legacy[1][0].shape[2], 4)
        self.assertEqual(legacy[0][0].shape[2], 0)

    def test_error_handling(self):
        k = np.ones((1, 1, 4, self.head_dim), dtype=np.float32)
        v = np.ones((1, 1, 4, self.head_dim), dtype=np.float32)

        with self.assertRaises(IndexError):
            self.cache.update(k, v, layer_idx=99)
        with self.assertRaises(ValueError):
            self.cache.update(np.ones((4, self.head_dim)), v, layer_idx=0) # 2D tensor
        with self.assertRaises(ValueError):
            self.cache.update(k, np.ones((1, 1, 4, 16)), layer_idx=0) # Mismatched dim


if __name__ == "__main__":
    unittest.main()
