---
name: mlx-development
description: >
  This skill should be used when the user asks to "add a new model",
  "implement KV caching", "modify the cache system", "add LoRA support",
  "extend generation", "add a sampling method", "implement a tool parser",
  "add quantization support", "modify the server API", "debug inference",
  "optimize memory usage", "implement sharding", "add distributed support",
  or modifies source modules under mlx_lm/. Covers MLX framework development
  patterns including model composition, KV cache architecture, generation
  pipeline, tuner/LoRA, quantization, sampling, server API, and tool parsing.
domains: [mlx, models, caching, generation, training, quantization]
---

## MLX-LM Development Patterns

This skill provides the architectural patterns, interfaces, and conventions
for developing within the mlx-lm codebase. MLX-LM is Apple's package for
running and fine-tuning LLMs on Apple Silicon using the MLX framework.

## Core Systems Overview

### KV Cache System (`mlx_lm/models/cache.py`)

The cache system is the performance-critical layer between models and
generation. All cache types share a common interface:

| Cache Type | Use Case | Key Property |
|---|---|---|
| `KVCache` | Default inference | Pre-allocates in 256-token steps |
| `RotatingKVCache` | Long sequences, memory-bound | Fixed-size ring buffer with `keep` tokens |
| `CompressedKVCache` | Intelligent eviction | L2-norm based key eviction with budget |
| `QuantizedKVCache` | Memory-constrained inference | Quantized KV storage (configurable bits) |
| `ConcatenateKVCache` | Simple concatenation | No pre-allocation |
| `ArraysCache` | Special attention patterns | List-based storage |
| `ChunkedKVCache` | Chunked attention windows | Fixed chunk_size |
| `BatchKVCache` | Batch generation | Wraps KVCache for multi-sequence |
| `BatchRotatingKVCache` | Batch + long context | Wraps RotatingKVCache for multi-sequence |

**Common interface** (all caches implement):
- `update_and_fetch(keys, values)` -> `(keys, values)` for attention
- `state` / `meta_state` properties for persistence
- `offset` property for RoPE position tracking
- `is_trimmable` / `empty` / `nbytes` properties

**Cache construction**: `make_prompt_cache(model, max_kv_size, compact_kv_budget)`
defers to `model.make_cache()` if available, otherwise creates `KVCache()`,
`RotatingKVCache(max_size, keep=4)`, or `CompressedKVCache(budget)` per layer.
`max_kv_size` and `compact_kv_budget` are mutually exclusive.

For complete cache class signatures, state management, mask generation, and
prompt caching patterns, see `references/kv-cache-system.md`.

### Model System (`mlx_lm/models/`)

Every model follows the canonical composition pattern:

```
@dataclass ModelArgs(BaseModelArgs) -> Attention -> MLP -> TransformerBlock -> Model
```

**ModelArgs**: Extends `BaseModelArgs` (provides `from_dict(cls, params)`).
Fields match HuggingFace `config.json` keys. Always a `@dataclass`.

**Attention**: Projects Q/K/V, applies RoPE via `initialize_rope()`, calls
`cache.update_and_fetch(keys, values)`, uses `scaled_dot_product_attention()`.

**MLP**: SwiGLU pattern with `gate_proj`, `down_proj`, `up_proj`.

**TransformerBlock**: Pre-norm residual connections using `nn.RMSNorm`.
Signature: `__call__(x, mask, cache)`.

**Model**: Wraps the inner model, applies `lm_head` or tied embeddings.
Implements `sanitize(weights)` for weight cleanup and `shard(group)` for
distributed inference. Optional `make_cache()` for custom cache strategies.

File naming: each file matches `model_type` from HuggingFace config
(e.g., `llama.py` for `"model_type": "llama"`). Aliases go in
`MODEL_REMAPPING` dict in `utils.py`.

For complete model implementation patterns, weight sanitization, sharding,
and the `BaseModelArgs` interface, see `references/model-patterns.md`.

### Generation Pipeline (`mlx_lm/generate.py`)

Three levels of generation API:

1. **`generate_step(prompt, model, *, max_tokens, sampler, logits_processors, ...)`**
   -- Low-level generator yielding `(token_id, logprobs)` tuples. Accepts
   pre-built `sampler` and `logits_processors` callables (not raw sampling
   params). Handles prefill in `prefill_step_size` chunks (default 2048),
   async eval with `mx.async_eval()`, and KV cache quantization via
   `maybe_quantize_kv_cache()`. Also accepts `compact_kv_budget` for
   L2-norm based cache compression via `maybe_compact_kv_cache()`.

2. **`stream_generate(model, tokenizer, prompt, ...)`** -- Mid-level generator
   yielding `GenerationResponse` dataclass objects with text, tokens, timing,
   and `finish_reason` ("length", "stop", or None).

3. **`generate(model, tokenizer, prompt, ...)`** -- High-level function
   collecting all tokens into a single string return.

**Batch generation**: `batch_generate()` and `BatchGenerator` class process
multiple prompts simultaneously using `BatchKVCache`. The `Batch` dataclass
tracks uids, tokens, caches, and per-sequence samplers.

**Speculative decoding**: `speculative_generate_step()` uses a draft model
for token candidates, returning `(token, logprobs, from_draft)` tuples.

For complete generation function signatures, `GenerationResponse` fields,
batch processing details, and speculative decoding, see
`references/generation-and-sampling.md`.

### Sampling System (`mlx_lm/sample_utils.py`)

**`make_sampler(temp, top_p, min_p, top_k, xtc_probability, ...)`** creates
a composable sampler chain: `top_p -> min_p -> xtc -> top_k -> categorical`.
At `temp=0`, returns `mx.argmax`. All sampling functions are `@mx.compile`
decorated for performance.

**`make_logits_processors(logit_bias, repetition_penalty, ...)`** returns a
list of `Callable[[mx.array, mx.array], mx.array]` functions applied before
sampling. Each takes `(tokens, logits)` and returns modified logits.

### Tuner/LoRA System (`mlx_lm/tuner/`)

**Extension point**: `linear_to_lora_layers(model, num_layers, config, use_dora)`
converts the last `num_layers` blocks' linear layers to LoRA equivalents.
Config dict: `{"rank": int, "scale": float, "dropout": float, "keys": list}`.

**LoRA layers** (`tuner/lora.py`): `LoRALinear`, `LoRAEmbedding`,
`LoRASwitchLinear` -- each with `from_base(base_layer, r, scale, dropout)`.

**Training** (`tuner/trainer.py`): `TrainingArgs` dataclass controls
`batch_size`, `iters`, `max_seq_length`, `adapter_file`, `grad_checkpoint`,
`grad_accumulation_steps`. Training loop uses `default_loss()` (cross-entropy
with masking) and `iterate_batches()` for length-sorted batching.

For LoRA config details, adapter loading/saving, and training loop internals,
see `references/tuner-and-quantization.md`.

### Quantization System (`mlx_lm/quant/`)

Four methods, each with own `main()` entry point routed via CLI:
- **AWQ** (`awq.py`): Per-layer scale configs, `ScaleConfig` dataclass
- **GPTQ** (`gptq.py`): Post-training quantization
- **DWQ** (`dwq.py`): Differentiable Weight Quantization
- **Dynamic** (`dynamic_quant.py`): Runtime quantization during inference

### Server System (`mlx_lm/server.py`)

OpenAI-compatible API server with `LRUPromptCache` for multi-turn caching.
`SearchResult` enum enables exact/shorter/longer prefix matching for cache
hits. Supports streaming via SSE, tool calling via pluggable parsers, and
multi-model loading.

### Tool Parsers (`mlx_lm/tool_parsers/`)

Pluggable function-calling parsers. Interface: `tool_call_start`/`tool_call_end`
strings plus `parse_tool_call(text, tools)` function. Implementations:
`json_tools`, `pythonic`, `mistral`, `qwen3_coder`, `function_gemma`,
`kimi_k2`, `glm47`, `longcat`, `minimax_m2`.

### Utilities (`mlx_lm/utils.py`)

- **`load(model_path, lazy, strict)`** -> `(model, tokenizer)`: Primary entry point
- **`MODEL_REMAPPING`** dict: Maps HuggingFace aliases to canonical implementations
- **`_get_classes(config)`** -> `(Model, ModelArgs)`: Dynamic model class resolution
- **`get_total_parameters(model)`**: Counts trainable/quantized params
- **`compute_bits_per_weight(model)`**: Memory efficiency metric

## Common Development Tasks

### Adding a New Model

1. Create `mlx_lm/models/<model_type>.py` matching HF `config.json` `model_type`
2. Implement: `ModelArgs(BaseModelArgs)`, `Attention`, `MLP`, `TransformerBlock`, `Model`
3. Implement `Model.sanitize(weights)` to clean HF weight names
4. Add `Model.shard(group)` for distributed support
5. If the model needs non-standard caching, add `Model.make_cache()`
6. Add LoRA support in `mlx_lm/tuner/utils.py` if needed
7. Add alias to `MODEL_REMAPPING` in `utils.py` if sharing an implementation
8. Add test in `tests/test_models.py`

### Modifying Cache Behavior

1. Extend `_BaseCache` or an existing cache class
2. Implement `update_and_fetch()`, `state`/`meta_state` properties
3. Wire into `make_prompt_cache()` or model's `make_cache()`
4. Verify mask generation works with `create_attention_mask()`

### Adding a Sampling Method

1. Create function in `sample_utils.py` with signature `(logprobs, ...) -> logprobs`
2. Decorate with `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`
3. Add to the sampler chain in `make_sampler()`
4. Add CLI argument in `generate.py:setup_arg_parser()`

## Additional Resources

### Reference Files

For detailed patterns and implementation guides, consult:
- **`references/kv-cache-system.md`** -- Complete cache class hierarchy, state
  management, mask generation, prompt caching, CompressedKVCache internals,
  and cross-layer coherent eviction
- **`references/model-patterns.md`** -- Model composition patterns, weight
  sanitization, distributed sharding, BaseModelArgs, and sliding window attention
- **`references/generation-and-sampling.md`** -- Generation pipeline internals,
  GenerationResponse fields, batch generation, speculative decoding, and
  sampler composition
- **`references/tuner-and-quantization.md`** -- LoRA/DoRA layer conversion,
  TrainingArgs, training loop, adapter persistence, and quantization methods
