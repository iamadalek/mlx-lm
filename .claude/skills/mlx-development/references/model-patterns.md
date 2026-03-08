# Model Implementation Patterns

Source: `mlx_lm/models/` (110+ model files)

## Canonical Model Structure

Every model file follows the same four-class composition. Use `llama.py`
as the canonical reference implementation.

### 1. ModelArgs (@dataclass)

```python
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    # Common optional fields:
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    sliding_window: Optional[int] = None
```

**BaseModelArgs** (`base.py`): Provides `from_dict(cls, params)` which
filters config.json keys to match dataclass fields via `inspect.signature`.

**Post-init defaults**: Use `__post_init__` for derived fields:
```python
def __post_init__(self):
    if self.num_key_value_heads is None:
        self.num_key_value_heads = self.num_attention_heads
```

### 2. Attention

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        # Linear projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)
        # RoPE initialization
        self.rope = initialize_rope(head_dim, rope_theta, ...)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape for multi-head: [B, L, D] -> [B, n_heads, L, head_dim]
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # RoPE + cache integration
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
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)
```

**Key patterns**:
- GQA (Grouped Query Attention): `n_kv_heads < n_heads` is standard
- `cache.offset` drives RoPE positional encoding
- `scale = head_dim ** -0.5` for attention scaling

### 3. MLP

Standard SwiGLU pattern:

```python
class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
```

`swiglu` is imported from `mlx_lm/models/activations.py`.

### 4. TransformerBlock

Pre-norm residual connections:

```python
class TransformerBlock(nn.Module):
    def __init__(self, args, use_sliding=False):
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(...)

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r
```

### 5. Inner Model (e.g., LlamaModel)

```python
class LlamaModel(nn.Module):
    def __init__(self, args):
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args, ...) for ...]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[self.fa_idx])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        return self.norm(h)
```

### 6. Outer Model (Model)

```python
class Model(nn.Module):
    def __init__(self, args):
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out
```

## Required Model Methods

### sanitize(weights)

Post-processes loaded HuggingFace weights:

```python
def sanitize(self, weights):
    # Remove precomputed rotary embeddings (computed dynamically)
    weights = {k: v for k, v in weights.items()
               if "self_attn.rotary_emb.inv_freq" not in k}
    # Remove tied weights
    if self.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
```

Common sanitization patterns:
- Remove `rotary_emb.inv_freq` (recomputed by MLX RoPE)
- Remove duplicate tied weights (`lm_head.weight` when tied)
- Rename weight keys to match MLX layer names
- Transpose weights where HF uses different layout

### shard(group)

Distributed inference support:

```python
def shard(self, group=None):
    group = group or mx.distributed.init()
    N = group.size()
    for layer in self.model.layers:
        # Attention: shard Q/K/V as all-to-sharded, O as sharded-to-all
        layer.self_attn.q_proj = shard_linear(
            layer.self_attn.q_proj, "all-to-sharded", group=group)
        layer.self_attn.o_proj = shard_linear(
            layer.self_attn.o_proj, "sharded-to-all", group=group)
        # MLP: shard gate/up as all-to-sharded, down as sharded-to-all
        layer.mlp.gate_proj = shard_linear(
            layer.mlp.gate_proj, "all-to-sharded", group=group)
        layer.mlp.down_proj = shard_linear(
            layer.mlp.down_proj, "sharded-to-all", group=group)
        # Update head counts
        layer.self_attn.n_heads //= N
        layer.self_attn.n_kv_heads //= N
```

Sharding strategy: fan-out layers use `all-to-sharded`, fan-in layers use
`sharded-to-all`.

### make_cache() (Optional)

Override default cache construction when the model needs heterogeneous caches:

```python
def make_cache(self):
    return [
        RotatingKVCache(max_size=self.model.sliding_window)
        if layer.use_sliding else KVCache()
        for layer in self.layers
    ]
```

### layers (Property)

All models must expose a `layers` property for cache construction and LoRA:

```python
@property
def layers(self):
    return self.model.layers
```

## Model Loading Pipeline

`utils.py` resolves models dynamically:

```python
def _get_classes(config):
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    # importlib.import_module(f"mlx_lm.models.{model_type}")
    # Returns (Model, ModelArgs) from the module
```

**MODEL_REMAPPING** maps aliases to canonical implementations:
```python
MODEL_REMAPPING = {
    "mistral": "llama",
    "llava": "mistral3",
    "kimi_k2": "deepseek_v3",
    # ...
}
```

## Sliding Window Attention

Models like Llama 3.x support mixed attention types via `layer_types`:

```python
@dataclass
class ModelArgs(BaseModelArgs):
    layer_types: Optional[List[str]] = None  # ["full_attention", "sliding_attention"]
    sliding_window: Optional[int] = None
```

Each layer type gets its own mask computed from `create_attention_mask()`.
The model's `make_cache()` returns `RotatingKVCache` for sliding layers
and `KVCache` for full attention layers.

## Variant Patterns

### MoE (Mixture of Experts)

Models like DeepSeek V2/V3 use expert routing:
- `SwitchLinear` for expert dispatch
- `LoRASwitchLinear` for LoRA on MoE layers

### Multi-Latent Attention (MLA)

Models like DeepSeek V2 use compressed KV representations:
- Custom cache types via `CacheList`
- Different head dimensions for Q vs KV

### Vision-Language Models

Models with image inputs:
- Separate vision encoder module
- `input_embeddings` parameter for pre-computed embeddings
- Additional preprocessing in `__call__`

## Weight Name Conventions

HuggingFace weight names typically follow:
```
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

Determine layer names from `model.safetensors.index.json` in HF repos
or from the Transformers source code.
