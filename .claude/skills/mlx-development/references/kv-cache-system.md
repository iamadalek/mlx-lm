# KV Cache System Deep Dive

Source: `mlx_lm/models/cache.py`

## Base Cache Interface

All caches inherit from `_BaseCache`:

```python
class _BaseCache:
    @property
    def state(self) -> Any:        # Serializable cache state (keys, values)
    @property
    def meta_state(self) -> Dict:  # Metadata (offset, type, config)
    @classmethod
    def from_state(cls, state, meta_state) -> _BaseCache:  # Reconstruct from saved
    def is_trimmable(self) -> bool: # Whether cache supports trimming
    def size(self) -> int:          # Sequence length (0 if not implemented)
    def empty(self) -> bool:        # Whether cache has been written to
    @property
    def nbytes(self) -> int:        # Memory consumed
```

## Cache Classes

### KVCache (Default)

Pre-allocates in 256-token steps for efficient appending:

```python
class KVCache(_BaseCache):
    step = 256
    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        # On first call or when step boundary exceeded: allocate new buffer
        # Copies new keys/values into pre-allocated space
        # Returns full key/value tensors up to current offset
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
```

**State format**: `(keys_array, values_array)` where shapes are
`[1, n_kv_heads, allocated_length, head_dim]`.

**Meta state**: Offset is reconstructed from `keys.shape[2]` on load.

**Additional methods**:
- `to_quantized(group_size, bits)` -> `QuantizedKVCache`: Converts to quantized
- `merge(caches)` -> `BatchKVCache`: Merges multiple caches for batching

### RotatingKVCache (Fixed-Size Ring Buffer)

Critical for long-context inference with bounded memory:

```python
class RotatingKVCache(_BaseCache):
    step = 256
    def __init__(self, max_size, keep=0):
        self.keep = keep        # Tokens preserved at start (never evicted)
        self.max_size = max_size  # Maximum ring buffer size
        self._idx = 0           # Current write position in ring
```

**How rotation works**:
1. First `keep` tokens are always preserved (e.g., system prompt)
2. Remaining slots form a ring buffer
3. When full, new KV pairs overwrite the oldest rotating entries
4. `_idx` tracks the current write position within the rotating region

**Two update paths**:
- `_update_in_place(keys, values)`: Single-token generation (S=1), writes
  directly to ring position, wraps `_idx` to `keep` when reaching `max_size`
- `_update_concat(keys, values)`: Multi-token prefill (S>1), puts cache in
  temporal order via `_temporal_order()`, trims oldest, concatenates new

**Mask generation**: `make_mask(N, window_size, return_array)` creates causal
masks that account for the ring buffer topology -- tokens in the "future"
positions of the ring are masked out. Handles both single-token (N=1) and
multi-token (N>1) cases with different masking strategies.

**Meta state**: `(keep, max_size, offset, _idx)` -- all needed to reconstruct
ring buffer position.

**Usage patterns**:
```python
# For models with sliding window attention:
RotatingKVCache(max_size=sliding_window_size)

# For memory-bounded long generation:
RotatingKVCache(max_size=4096, keep=4)
```

### CompressedKVCache (L2-Norm Based Eviction)

Intelligent cache compression that evicts tokens with lowest L2-norm keys:

```python
class CompressedKVCache(_BaseCache):
    step = 256
    def __init__(self, budget: int, keep_recent: int = 32):
        self.budget = budget            # Max tokens after compaction
        self.keep_recent = keep_recent  # Recent tokens protected from eviction
        self._physical_idx = 0          # Actual position in buffer
        self.offset = 0                 # Logical position (for RoPE, never decremented)
```

**Key invariant**: `offset` tracks the true sequence position for RoPE and is
NEVER decremented during compaction. `_physical_idx` tracks actual buffer
position and is reset after eviction. This separation is critical -- RoPE
needs absolute positions, but the physical buffer shrinks.

**Eviction heuristic**: Tokens with large L2-norm keys attract more attention
(attention is proportional to Q-K dot product). High-norm keys correspond to
"attention sinks" (BOS, punctuation) whose removal causes attention collapse.
By keeping high-norm keys and evicting low-norm ones, critical tokens for
attention stability are retained.

**Compaction methods**:
- `compact(kept_indices)`: Compacts using pre-computed indices (for cross-layer
  coherent eviction)
- `_compute_kept_indices(active_keys)`: Computes kept indices from this layer's
  keys alone (single-layer eviction)
- `indices_from_norms(norms)`: Public entry point for cross-layer eviction --
  given aggregated norms `(B, seq_len)`, returns sorted kept indices

**Cross-layer coherent eviction** via `maybe_compact_kv_cache()` in
`generate.py`:
1. Aggregates L2 norms across ALL CompressedKVCache layers
2. Computes a single set of kept indices from aggregated norms
3. Applies same indices to every layer (same tokens kept everywhere)
4. Uses hysteresis: triggers only when size > budget + max(keep_recent, 64)

**Mask generation**: Uses `_physical_idx` (not `offset`) because the mask
covers physical cache slots, not logical positions.

**Constraints**:
- B>1 not supported (scalar offset can't represent per-batch RoPE positions)
- `budget` must be > `keep_recent`
- `is_trimmable()` returns True only when offset == _physical_idx (before any
  compaction has occurred)

**Usage**:
```python
# Via make_prompt_cache:
cache.make_prompt_cache(model, compact_kv_budget=2048)

# Via generate_step:
generate_step(prompt, model, compact_kv_budget=2048)

# Via CLI:
mlx_lm generate --compact-kv-budget 2048
```

### QuantizedKVCache

Reduces KV memory via quantization during inference:

```python
class QuantizedKVCache(_BaseCache):
    step = 256
    def __init__(self, group_size: int = 64, bits: int = 8):
        # Stores quantized keys/values instead of full precision
```

**Activation**: Called via `maybe_quantize_kv_cache()` in `generate.py`
when `quantized_kv_start` threshold is exceeded:

```python
def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)
```

**State format**: Stores `(quantized_keys, scales, biases, quantized_values, ...)`.

**Meta state**: `(offset, group_size, bits)`.

### BatchKVCache and BatchRotatingKVCache

Wrappers for batch generation that manage left-padded multi-sequence caches:

```python
class BatchKVCache(_BaseCache):
    def __init__(self, left_padding: List[int]):
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])  # Per-sequence offsets
        self._idx = 0  # Shared write position
```

**Left-padding**: Prompts of different lengths are left-padded to align:
```
[0, 1, 3, 5]   # padding=1
[0, 0, 0, 7]   # padding=3
[2, 6, 8, 9]   # padding=0
```

**Batch operations**:
- `filter(batch_indices)`: In-place removal of finished sequences
- `extend(other)`: In-place merge of new batch into active batch
- `extract(idx)`: Extract single-sequence KVCache from batch
- `prepare(left_padding, lengths, right_padding)`: Setup for right-padded inputs
- `finalize()`: Converts right-padded back to left-padded after prefill

**Merge**: `BatchKVCache.merge(caches)` takes a list of single-sequence
KVCache objects and creates a batched cache with appropriate padding.

### CacheList

Aggregates heterogeneous caches per layer:

```python
class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = caches  # Tuple of cache objects for a single layer
```

Used when a layer needs multiple cache types (e.g., MLA attention).
Supports `filter`, `extend`, `extract`, `prepare`, `finalize` by
delegating to each sub-cache.

### ArraysCache

List-based cache for non-standard attention patterns (e.g., Mamba, RWKV):

```python
class ArraysCache(_BaseCache):
    def __init__(self, size, left_padding=None):
        self.cache = [None] * size
```

Supports left-padding masks, batch filtering/extending, and length tracking.

### ChunkedKVCache

Maintains fixed-size attention windows with front trimming:

```python
class ChunkedKVCache(_BaseCache):
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.start_position = 0

    def maybe_trim_front(self):
        # Trims cache when it exceeds chunk_size
```

## Cache Construction

### make_prompt_cache()

Entry point for cache creation:

```python
def make_prompt_cache(model, max_kv_size=None, compact_kv_budget=None):
    # max_kv_size and compact_kv_budget are mutually exclusive
    if hasattr(model, "make_cache"):
        return model.make_cache()   # Model-specific cache strategy
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size, keep=4) ...]
    elif compact_kv_budget is not None:
        return [CompressedKVCache(budget=compact_kv_budget) ...]
    else:
        return [KVCache() ...]
```

**Custom make_cache() examples**:

Llama with sliding window attention:
```python
def make_cache(self):
    return [
        RotatingKVCache(max_size=self.model.sliding_window)
        if layer.use_sliding else KVCache()
        for layer in self.layers
    ]
```

### Prompt Cache Persistence

Save and load pre-computed caches for reuse:

```python
save_prompt_cache(file_name, cache, metadata={})
# Saves cache.state + cache.meta_state to .safetensors

load_prompt_cache(file_name, return_metadata=False)
# Reconstructs cache objects from saved state
# Uses globals()[class_name].from_state() for deserialization
```

## Cache Integration with Models

### In Attention Layers

```python
class Attention(nn.Module):
    def __call__(self, x, mask=None, cache=None):
        # ... project Q, K, V ...
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
```

Key points:
- `cache.offset` provides the position for RoPE encoding
- `cache.update_and_fetch()` appends new KV pairs and returns the full history
- `scaled_dot_product_attention` receives the cache for potential mask delegation

### Mask Generation

`create_attention_mask()` in `cache.py` (module-level function):

```python
def create_attention_mask(N, offset, return_array, window_size):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None        # Single-token generation needs no mask
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"    # String sentinel for SDPA fast path
```

Each cache class implements `make_mask()` which delegates to this function
with the appropriate offset. The `"causal"` string return is an optimization
-- `scaled_dot_product_attention` uses a fast path for simple causal masks.

## Memory Management Patterns

### Periodic Cache Cleanup

In `generate_step()`, memory is managed every 256 tokens:

```python
if n % 256 == 0:
    mx.clear_cache()
```

### Wired Memory Limit

For large models, `wired_limit()` context manager adjusts macOS memory:

```python
with wired_limit(model, [generation_stream]):
    # Generation with adjusted iogpu.wired_limit_mb
```

## Trimming and Modification

Caches that support trimming implement `is_trimmable()` -> True and `trim(n)`:

```python
def trim_prompt_cache(cache, num_tokens):
    if not can_trim_prompt_cache(cache):
        return 0
    return [c.trim(num_tokens) for c in cache][0]
```

This is used by the server's `LRUPromptCache` to rewind caches to a
common prefix when reusing cached prompts across requests.

**Trimmability rules**:
- `KVCache`: Always trimmable
- `RotatingKVCache`: Trimmable only when offset < max_size
- `CompressedKVCache`: Trimmable only when offset == _physical_idx (no compaction yet)
- `QuantizedKVCache`: Always trimmable
- `ConcatenateKVCache`: Always trimmable
