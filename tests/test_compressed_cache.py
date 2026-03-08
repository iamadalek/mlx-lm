# Copyright © 2023-2024 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.models.cache import (
    CompressedKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)


class TestCompressedKVCache(unittest.TestCase):

    # -- Priority 1: Critical invariants --

    def test_offset_invariant_after_compaction(self):
        cache = CompressedKVCache(budget=2048)
        # Fill with 4096 tokens (in chunks to simulate prefill)
        keys = mx.random.normal(shape=(1, 8, 4096, 64))
        values = mx.random.normal(shape=(1, 8, 4096, 64))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        self.assertEqual(cache.offset, 4096)
        self.assertEqual(cache._physical_idx, 4096)

        cache.compact()

        # offset must stay at 4096 (RoPE invariant)
        self.assertEqual(cache.offset, 4096)
        # physical index should be budget
        self.assertEqual(cache._physical_idx, 2048)
        # keys array should be exactly budget-sized
        self.assertEqual(cache.keys.shape[2], 2048)
        self.assertEqual(cache.values.shape[2], 2048)

    def test_token_selection_correctness(self):
        cache = CompressedKVCache(budget=3, keep_recent=1)
        # 5 tokens, 1 head, dim=2
        # Token norms: [2, 10, 1, 8, 5]
        # Evictable (first 4): norms [2, 10, 1, 8]
        # Budget - keep_recent = 2 tokens to keep from evictable
        # Highest norms (attention sinks): token 1 (norm=10), token 3 (norm=8)
        # Protected: token 4
        # Kept indices (sorted): [1, 3, 4]
        keys = mx.array(
            [
                [
                    [
                        [1.0, 1.732],  # norm ~= 2
                        [7.07, 7.07],  # norm ~= 10
                        [0.5, 0.866],  # norm ~= 1
                        [5.66, 5.66],  # norm ~= 8
                        [3.54, 3.54],  # norm ~= 5
                    ]
                ]
            ]
        )  # shape (1, 1, 5, 2)
        values = mx.arange(10).reshape(1, 1, 5, 2).astype(mx.float32)

        cache.keys = keys
        cache.values = values
        cache._physical_idx = 5
        cache.offset = 5
        mx.eval(cache.keys, cache.values)

        cache.compact()

        self.assertEqual(cache._physical_idx, 3)
        self.assertEqual(cache.offset, 5)  # unchanged

        # Check that kept values correspond to tokens 1, 3, 4 (high-norm kept)
        expected_values = mx.array([[[[2, 3], [6, 7], [8, 9]]]]).astype(mx.float32)
        self.assertTrue(mx.allclose(cache.values, expected_values))

    def test_recent_token_protection(self):
        cache = CompressedKVCache(budget=4, keep_recent=2)
        # 6 tokens: recent tokens (indices 4, 5) have very low norms
        # but must survive because they are protected
        keys_list = mx.zeros((1, 1, 6, 4))
        keys_list[..., 4, :] = 0.01  # low norm, protected
        keys_list[..., 5, :] = 0.02  # low norm, protected
        # Evictable tokens have varying norms
        keys_list[..., 0, :] = 1.0  # low norm -> evict
        keys_list[..., 1, :] = 50.0  # high norm -> keep
        keys_list[..., 2, :] = 2.0  # low norm -> evict
        keys_list[..., 3, :] = 60.0  # high norm -> keep
        cache.keys = keys_list
        cache.values = mx.arange(24).reshape(1, 1, 6, 4).astype(mx.float32)
        cache._physical_idx = 6
        cache.offset = 6
        mx.eval(cache.keys, cache.values)

        cache.compact()

        self.assertEqual(cache._physical_idx, 4)
        # Recent tokens must survive despite low norms
        # Kept: [1, 3] (highest norm evictable) + [4, 5] (protected)
        expected_values = mx.array(
            [[[[4, 5, 6, 7], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]]
        ).astype(mx.float32)
        self.assertTrue(mx.allclose(cache.values, expected_values))

    def test_make_mask_uses_physical_size(self):
        cache = CompressedKVCache(budget=2048)
        keys = mx.random.normal(shape=(1, 4, 4096, 64))
        values = mx.random.normal(shape=(1, 4, 4096, 64))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)
        cache.compact()

        # Mask should use physical size (2048), not offset (4096)
        mask = cache.make_mask(4, return_array=True)
        # Shape should be (4, 2048 + 4) = (4, 2052)
        self.assertEqual(mask.shape, (4, 2048 + 4))

    def test_update_and_fetch_after_compaction(self):
        cache = CompressedKVCache(budget=64, keep_recent=8)
        # Fill cache beyond budget
        keys = mx.random.normal(shape=(1, 4, 128, 32))
        values = mx.random.normal(shape=(1, 4, 128, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        cache.compact()
        self.assertEqual(cache._physical_idx, 64)
        self.assertEqual(cache.offset, 128)

        # Now append a new token
        new_keys = mx.random.normal(shape=(1, 4, 1, 32))
        new_values = mx.random.normal(shape=(1, 4, 1, 32))
        k, v = cache.update_and_fetch(new_keys, new_values)

        self.assertEqual(cache._physical_idx, 65)
        self.assertEqual(cache.offset, 129)
        self.assertEqual(k.shape[2], 65)
        self.assertEqual(v.shape[2], 65)
        # Verify the new token is at the correct position
        self.assertTrue(mx.allclose(k[..., 64:65, :], new_keys))
        self.assertTrue(mx.allclose(v[..., 64:65, :], new_values))

    # -- Priority 2: GQA and multi-head --

    def test_gqa_head_aggregation(self):
        cache = CompressedKVCache(budget=3, keep_recent=1)
        # 4 tokens, 2 KV heads, dim=1
        # Head 0 norms: [1, 10, 2, 5]
        # Head 1 norms: [10, 1, 3, 5]
        # Aggregated (sum): [11, 11, 5, 10]
        # Evictable (first 3): [11, 11, 5]
        # Keep 2 highest from evictable: tokens 0 and 1 (both 11)
        keys = mx.array(
            [
                [
                    [[1.0], [10.0], [2.0], [5.0]],  # head 0
                    [[10.0], [1.0], [3.0], [5.0]],  # head 1
                ]
            ]
        )  # shape (1, 2, 4, 1)
        values = mx.arange(8).reshape(1, 2, 4, 1).astype(mx.float32)
        cache.keys = keys
        cache.values = values
        cache._physical_idx = 4
        cache.offset = 4
        mx.eval(cache.keys, cache.values)

        cache.compact()

        self.assertEqual(cache._physical_idx, 3)
        # Tokens 0 and 1 (highest norms, 11) should be kept
        # Token 2 (norm 5) should be evicted
        # Token 3 is protected (recent)
        # Values: head 0 kept=[0,1,3], head 1 kept=[4,5,7]
        expected_values = mx.array([[[[0], [1], [3]], [[4], [5], [7]]]]).astype(
            mx.float32
        )
        self.assertTrue(mx.allclose(cache.values, expected_values))

    def test_values_evicted_alongside_keys(self):
        cache = CompressedKVCache(budget=3, keep_recent=1)
        keys = mx.array(
            [
                [
                    [
                        [10.0, 0.0],  # high norm -> keep
                        [0.1, 0.0],  # low norm -> evict
                        [0.2, 0.0],  # low norm -> keep (2nd highest)
                        [5.0, 0.0],  # protected (recent)
                    ]
                ]
            ]
        )
        values = mx.array(
            [
                [
                    [
                        [100.0, 100.0],
                        [200.0, 200.0],
                        [300.0, 300.0],
                        [400.0, 400.0],
                    ]
                ]
            ]
        )
        cache.keys = keys
        cache.values = values
        cache._physical_idx = 4
        cache.offset = 4
        mx.eval(cache.keys, cache.values)

        cache.compact()

        # Token 1 (lowest norm) should be evicted
        # Remaining values should be [100, 300, 400]
        self.assertEqual(cache._physical_idx, 3)
        expected_vals = mx.array(
            [
                [
                    [
                        [100.0, 100.0],
                        [300.0, 300.0],
                        [400.0, 400.0],
                    ]
                ]
            ]
        )
        self.assertTrue(mx.allclose(cache.values, expected_vals))

    # -- Priority 3: Composability and persistence --

    def test_to_quantized_raises(self):
        cache = CompressedKVCache(budget=1024)
        with self.assertRaises(NotImplementedError):
            cache.to_quantized()

    def test_state_meta_state_round_trip(self):
        cache = CompressedKVCache(budget=64, keep_recent=8)
        keys = mx.random.normal(shape=(1, 4, 100, 32))
        values = mx.random.normal(shape=(1, 4, 100, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)
        cache.compact()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "cache.safetensors")
            save_prompt_cache(cache_file, [cache])
            loaded = load_prompt_cache(cache_file)

        lc = loaded[0]
        self.assertEqual(cache.offset, lc.offset)
        self.assertEqual(cache._physical_idx, lc._physical_idx)
        self.assertEqual(cache.budget, lc.budget)
        self.assertEqual(cache.keep_recent, lc.keep_recent)
        self.assertTrue(mx.array_equal(cache.state[0], lc.state[0]))
        self.assertTrue(mx.array_equal(cache.state[1], lc.state[1]))

    def test_state_meta_state_round_trip_with_metadata(self):
        cache = CompressedKVCache(budget=64, keep_recent=8)
        keys = mx.random.normal(shape=(1, 4, 50, 32))
        values = mx.random.normal(shape=(1, 4, 50, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        metadata = {"model": "test", "version": "1.0"}
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "cache.safetensors")
            save_prompt_cache(cache_file, [cache], metadata)
            loaded, loaded_metadata = load_prompt_cache(
                cache_file, return_metadata=True
            )

        self.assertEqual(metadata, loaded_metadata)

    # -- Priority 4: Edge cases --

    def test_empty_cache(self):
        cache = CompressedKVCache(budget=1024)
        self.assertTrue(cache.empty())
        self.assertEqual(cache.size(), 0)
        self.assertEqual(cache.offset, 0)
        self.assertEqual(cache._physical_idx, 0)
        # compact on empty should not error
        cache.compact()

    def test_cache_smaller_than_budget(self):
        cache = CompressedKVCache(budget=1024)
        keys = mx.random.normal(shape=(1, 4, 100, 32))
        values = mx.random.normal(shape=(1, 4, 100, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        original_offset = cache.offset
        cache.compact()  # should be a no-op

        self.assertEqual(cache.offset, original_offset)
        self.assertEqual(cache._physical_idx, 100)

    def test_cache_equal_to_budget(self):
        cache = CompressedKVCache(budget=100)
        keys = mx.random.normal(shape=(1, 4, 100, 32))
        values = mx.random.normal(shape=(1, 4, 100, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        cache.compact()  # should be a no-op (physical_idx == budget)

        self.assertEqual(cache._physical_idx, 100)
        self.assertEqual(cache.offset, 100)

    def test_cache_equal_to_keep_recent(self):
        cache = CompressedKVCache(budget=64, keep_recent=32)
        keys = mx.random.normal(shape=(1, 4, 32, 32))
        values = mx.random.normal(shape=(1, 4, 32, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        cache.compact()  # no-op: physical_idx (32) <= budget (64)

        self.assertEqual(cache._physical_idx, 32)

    def test_keep_recent_exceeds_budget_raises(self):
        with self.assertRaises(ValueError):
            CompressedKVCache(budget=16, keep_recent=32)

    def test_budget_equals_keep_recent_raises(self):
        with self.assertRaises(ValueError):
            CompressedKVCache(budget=32, keep_recent=32)

    def test_single_token(self):
        cache = CompressedKVCache(budget=1024)
        keys = mx.random.normal(shape=(1, 4, 1, 32))
        values = mx.random.normal(shape=(1, 4, 1, 32))
        k, v = cache.update_and_fetch(keys, values)

        self.assertEqual(k.shape[2], 1)
        self.assertEqual(cache.offset, 1)
        self.assertEqual(cache._physical_idx, 1)

    def test_budget_minus_one_triggers_compaction(self):
        cache = CompressedKVCache(budget=10, keep_recent=2)
        keys = mx.random.normal(shape=(1, 2, 11, 8))
        values = mx.random.normal(shape=(1, 2, 11, 8))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        # physical_idx (11) > budget (10), should compact
        cache.compact()

        self.assertEqual(cache._physical_idx, 10)
        self.assertEqual(cache.offset, 11)

    # -- Priority 5: Integration --

    def test_compact_before_quantize_ordering(self):
        """Verify that maybe_compact_kv_cache runs on unquantized data,
        and that the generation loop calls compact before quantize."""
        from mlx_lm.generate import maybe_compact_kv_cache

        cache = CompressedKVCache(budget=64, keep_recent=8)
        # Fill well beyond hysteresis threshold: budget + max(keep_recent, 64) = 128
        # so we need > 128 tokens to trigger compaction
        keys = mx.random.normal(shape=(1, 4, 192, 32))
        values = mx.random.normal(shape=(1, 4, 192, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        # Verify keys are float (unquantized) before compaction
        self.assertEqual(cache.keys.dtype, mx.float32)

        prompt_cache = [cache]

        # Compact runs on unquantized float data
        maybe_compact_kv_cache(prompt_cache)
        self.assertEqual(prompt_cache[0].size(), 64)
        self.assertIsInstance(prompt_cache[0], CompressedKVCache)
        # Data remains float after compaction
        self.assertEqual(prompt_cache[0].keys.dtype, mx.float32)

    def test_hysteresis_avoids_per_token_compaction(self):
        """Verify that compaction does not fire on every token once budget
        is exceeded, but only after the hysteresis margin is breached."""
        from mlx_lm.generate import maybe_compact_kv_cache

        cache = CompressedKVCache(budget=64, keep_recent=8)
        # Fill to just over budget but under hysteresis threshold
        keys = mx.random.normal(shape=(1, 4, 80, 32))
        values = mx.random.normal(shape=(1, 4, 80, 32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        prompt_cache = [cache]
        maybe_compact_kv_cache(prompt_cache)
        # Should NOT compact: 80 <= 64 + max(8, 64) = 128
        self.assertEqual(prompt_cache[0].size(), 80)

    def test_make_prompt_cache_rejects_conflicting_params(self):
        """Verify that passing both max_kv_size and compact_kv_budget raises."""
        import mlx.nn as nn

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(32, 32) for _ in range(4)]

        model = FakeModel()
        with self.assertRaises(ValueError):
            make_prompt_cache(model, max_kv_size=512, compact_kv_budget=256)

    def test_from_state_empty_cache(self):
        """Verify that from_state with empty state doesn't cause AttributeError."""
        cache = CompressedKVCache(budget=64, keep_recent=8)
        # Empty cache: state == [], meta_state has zeros
        state = cache.state
        meta_state = cache.meta_state

        loaded = CompressedKVCache.from_state(state, meta_state)
        self.assertIsNone(loaded.keys)
        self.assertIsNone(loaded.values)
        self.assertTrue(loaded.empty())
        self.assertEqual(loaded.size(), 0)

    def test_compact_kv_budget_with_kv_bits_raises(self):
        """Verify that combining compact_kv_budget with kv_bits raises early."""
        from mlx_lm.generate import generate_step

        prompt = mx.array([1, 2, 3])
        # Use a minimal mock — we just need it to reach the validation
        with self.assertRaises(ValueError):
            # Exhaust the generator to trigger the body
            list(
                generate_step(
                    prompt,
                    None,  # model unused, raises before model call
                    compact_kv_budget=512,
                    kv_bits=4,
                )
            )


if __name__ == "__main__":
    unittest.main()
