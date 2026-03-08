# Generation Pipeline and Sampling

Source: `mlx_lm/generate.py`, `mlx_lm/sample_utils.py`

## Generation API Layers

### generate_step() -- Low-Level Generator

```python
def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    compact_kv_budget: Optional[int] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
```

**Important**: `generate_step()` does NOT accept raw sampling parameters
(`temp`, `top_p`, `top_k`, etc.) directly. Construct a `sampler` via
`make_sampler()` and `logits_processors` via `make_logits_processors()`
before passing them in.

**Cache strategies** (mutually exclusive):
- `max_kv_size`: Creates `RotatingKVCache` for bounded memory
- `compact_kv_budget`: Creates `CompressedKVCache` with L2-norm eviction
- `kv_bits` + `quantized_kv_start`: Quantizes existing cache after threshold

**Prefill phase**: Processes prompt in `prefill_step_size` chunks:
```python
while total_prompt_tokens - prompt_processed_tokens > 1:
    remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
    n_to_process = min(prefill_step_size, remaining)
    _model_call(input_tokens=prompt[:n_to_process][None], ...)
    compact_cache_fn(prompt_cache)
    quantize_cache_fn(prompt_cache)
    mx.eval([c.state for c in prompt_cache])
    prompt = prompt[n_to_process:]
    mx.clear_cache()
```

**Generation loop**:
```python
while True:
    logits = model(y[None], cache=prompt_cache)
    logits = logits[:, -1, :]
    # Apply logits processors
    for processor in logits_processors:
        logits = processor(tokens, logits)
    # Compact and quantize caches
    compact_cache_fn(prompt_cache)
    quantize_cache_fn(prompt_cache)
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    y = sampler(logprobs)
    yield y.item(), logprobs
```

**Async evaluation**: Uses `mx.async_eval()` for overlapped compute:
```python
mx.async_eval(y, logprobs)  # Start computing next token while yielding current
```

### stream_generate() -- Mid-Level Generator

```python
def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    **kwargs,  # All generate_step kwargs including compact_kv_budget
) -> Generator[GenerationResponse, None, None]:
```

Wraps `generate_step()` with tokenization/detokenization and timing:

```python
@dataclass
class GenerationResponse:
    text: str                        # Decoded text so far
    token: int                       # Current token ID
    logprobs: mx.array               # Log probabilities
    from_draft: bool                 # True if from speculative decoding
    prompt_tokens: int               # Number of prompt tokens
    prompt_tps: float                # Prompt processing speed (tokens/sec)
    generation_tokens: int           # Number of generated tokens
    generation_tps: float            # Generation speed (tokens/sec)
    peak_memory: float               # Peak memory in GB
    finish_reason: Optional[str]     # "length", "stop", or None
```

**Stop conditions**: Generation stops when:
1. `max_tokens` reached -> `finish_reason = "length"`
2. EOS token generated -> `finish_reason = "stop"`
3. Detokenizer produces complete output

**Note**: `compact_kv_budget` is not supported with speculative decoding
(a warning is emitted and the parameter is ignored).

### generate() -- High-Level Function

```python
def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
```

Collects all tokens and returns the full generated string. With
`verbose=True`, prints tokens and timing information. Remaining kwargs
are passed through to `stream_generate()`.

## Batch Generation

### batch_generate()

```python
def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    logits_processors: Optional[List[Callable]] = None,
    **kwargs,
) -> BatchResponse:
```

Returns `BatchResponse(texts, stats, caches)`.

### BatchGenerator Class

Manages concurrent generation across multiple prompts:

```python
@dataclass
class Batch:
    uids: List[int]                 # Unique IDs per sequence
    y: mx.array                     # Current tokens [batch, 1]
    logprobs: mx.array              # Current logprobs
    max_tokens: List[int]           # Per-sequence limits
    num_tokens: List[int]           # Generated count per sequence
    cache: List[Any]                # Per-layer batch caches
    samplers: List[Any]             # Per-sequence samplers
    logits_processors: List[Any]    # Per-sequence logits processors
    tokens: List[mx.array]          # Token history per sequence
```

**Two-phase batching**:
1. **Prefill phase**: Processes prompts with `prefill_batch_size`
2. **Completion phase**: Generates tokens with `completion_batch_size`

Sequences complete independently; finished sequences are removed from the
active batch via `Batch.filter()`.

**BatchGenerator features**:
- `insert(prompts, max_tokens, caches, samplers, ...)`: Add new prompts
- `remove(uids, return_prompt_caches)`: Remove sequences mid-generation
- `next()`: Process one generation step, returns `List[Response]`
- `stats()`: Returns `BatchStats` with timing and throughput

## Speculative Decoding

```python
def speculative_generate_step(
    prompt, model, draft_model,
    *, num_draft_tokens=2, max_tokens=256,
    sampler=None, logits_processors=None,
    prompt_cache=None, prefill_step_size=512,
    kv_bits=None, kv_group_size=64, quantized_kv_start=0,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
```

Uses a smaller draft model to propose `num_draft_tokens` candidate tokens,
then verifies with the main model. Yields `(token, logprobs, from_draft)`.

## Sampling System

### Sampler Construction

```python
make_sampler(
    temp=0.0, top_p=0.0, min_p=0.0, min_tokens_to_keep=1,
    top_k=0, xtc_probability=0.0, xtc_threshold=0.0,
    xtc_special_tokens=[],
) -> Callable[[mx.array], mx.array]
```

**Chain order**: `top_p -> min_p -> xtc -> top_k -> categorical_sampling`

At `temp=0`: returns `lambda x: mx.argmax(x, axis=-1)`.

### Sampling Functions

All decorated with `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`:

| Function | Signature | Description |
|---|---|---|
| `apply_top_p(logprobs, top_p)` | Nucleus sampling | Keeps tokens above cumulative prob threshold |
| `apply_min_p(logprobs, min_p, min_tokens_to_keep)` | Min-p filtering | Filters below scaled minimum probability |
| `apply_top_k(logprobs, top_k)` | Top-k sampling | Restricts to k highest-probability tokens |
| `apply_xtc(logits, prob, threshold, special_tokens)` | XTC sampling | Probabilistic extreme token choice |
| `categorical_sampling(logits, temp)` | Temperature sampling | `mx.random.categorical(logits * (1/temp))` |

### Logits Processors

```python
make_logits_processors(
    logit_bias=None,
    repetition_penalty=None,
    repetition_context_size=20,
) -> List[Callable[[mx.array, mx.array], mx.array]]
```

Each processor signature: `(tokens: mx.array, logits: mx.array) -> mx.array`

**logit_bias_processor**: Adds bias values to specific token indices.

**repetition_penalty**: Penalizes repeated tokens in the last
`repetition_context_size` positions. Positive logits are divided by penalty,
negative logits are multiplied.

### Adding a New Sampling Method

1. Implement function with signature `(logprobs: mx.array, ...) -> mx.array`
2. Apply `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`
3. Add to chain in `make_sampler()` at appropriate position
4. Add parameter to `make_sampler()` signature
5. Wire through CLI in `generate.py:setup_arg_parser()` and defaults

## Server Integration

The server (`server.py`) uses `generate_step()` directly with its own
`LRUPromptCache` for multi-turn caching:

```python
class LRUPromptCache:
    class SearchResult(Enum):
        EXACT = 0    # Exact prefix match
        SHORTER = 1  # Cached prefix is shorter than request
        LONGER = 2   # Cached prefix is longer (trim needed)

    def _search(self, prompt_tokens):
        # Finds best matching cached prompt prefix
        # Returns (SearchResult, cache, matched_length)
```

The server builds `stopping_criteria()` from EOS tokens and stop sequences,
and uses `sequence_overlap()` to handle partial stop sequence matches at
chunk boundaries during streaming.
