# Tuner/LoRA and Quantization Systems

Source: `mlx_lm/tuner/`, `mlx_lm/quant/`

## LoRA/DoRA Layer System

### Extension Point: linear_to_lora_layers()

```python
def linear_to_lora_layers(
    model: nn.Module,
    num_layers: int,       # Convert last N transformer blocks
    config: Dict,          # {"rank": int, "scale": float, "dropout": float, "keys": list}
    use_dora: bool = False,
):
```

**Config dict fields**:
- `rank` (int): Low-rank dimension (commonly 8-64)
- `scale` (float): LoRA scaling factor (commonly 1.0-20.0)
- `dropout` (float): Dropout on LoRA paths (commonly 0.0-0.1)
- `keys` (list, optional): Specific layer name patterns to convert. If None,
  auto-discovers all convertible layers.

**Auto-discovery**: When `keys` is None, walks the last `num_layers` blocks
and finds all layers matching:
- `nn.Linear`, `nn.QuantizedLinear`
- `SwitchLinear`, `QuantizedSwitchLinear`
- `nn.Embedding`, `nn.QuantizedEmbedding`

**Conversion dispatch**:
```python
# Layer -> LoRA equivalent:
nn.Linear / nn.QuantizedLinear  -> LoRALinear / DoRALinear
SwitchLinear / QuantizedSwitch  -> LoRASwitchLinear
nn.Embedding / nn.QuantizedEmb -> LoRAEmbedding / DoRAEmbedding
```

### LoRA Layer Implementations (`tuner/lora.py`)

#### LoRALinear

```python
class LoRALinear(nn.Module):
    @classmethod
    def from_base(cls, linear, r=8, scale=1.0, dropout=0.0):
        # Wraps existing linear with low-rank A/B matrices
        # output = base_linear(x) + dropout(B @ A @ x) * scale

    def fuse(self):
        # Merges LoRA weights into base weight for inference
        # Returns standard nn.Linear
```

#### LoRAEmbedding

```python
class LoRAEmbedding(nn.Module):
    @classmethod
    def from_base(cls, embedding, r=8, scale=1.0, dropout=0.0):
        # LoRA on embedding lookup tables
```

#### LoRASwitchLinear

```python
class LoRASwitchLinear(nn.Module):
    # LoRA for MoE expert routing layers
    # Used with DeepSeek and similar architectures
```

### DoRA Layers (`tuner/dora.py`)

Diagonal Output Rank Adaptation -- same interface as LoRA:
- `DoRALinear.from_base(linear, r, scale, dropout)`
- `DoRAEmbedding.from_base(embedding, r, scale, dropout)`

DoRA adds a learned magnitude vector to the LoRA decomposition.

### Adapter Persistence

```python
# Save adapters
def save_adapters(model, adapter_file):
    # Saves only LoRA/DoRA parameters to .safetensors

# Load adapters
def load_adapters(model, adapter_file):
    # Loads adapter weights and applies to model
    # Sets LoRA layers to non-trainable base + trainable adapter

# Remove adapters
def remove_lora_layers(model):
    # Reverts LoRA layers to base Linear/Embedding
    # Used when fusing or switching adapters
```

### Trainable Parameter Reporting

```python
def print_trainable_parameters(model):
    # Shows: "Trainable parameters: X / Y (Z%)"
    # LoRA typically makes 0.1-2% of parameters trainable
```

## Training System (`tuner/trainer.py`)

### TrainingArgs

```python
@dataclass
class TrainingArgs:
    batch_size: int = 4
    iters: int = 100
    val_batches: int = 25           # -1 for entire validation set
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    max_seq_length: int = 2048
    adapter_file: str = "adapters.safetensors"
    grad_checkpoint: bool = False   # Trade compute for memory
    grad_accumulation_steps: int = 1
```

### Loss Function

```python
def default_loss(model, batch, lengths):
    inputs = batch[:, :-1]     # All tokens except last
    targets = batch[:, 1:]     # All tokens except first
    logits = model(inputs)
    # Cross-entropy with length-based masking
    # Masks padding tokens in variable-length batches
```

### Training Loop

```python
def train(model, tokenizer, optimizer, train_set, val_set, args, loss):
    # Core loop:
    for batch in iterate_batches(train_set, batch_size, max_seq_length):
        loss_value, grads = loss_value_and_grad(model, batch, lengths)
        # Gradient accumulation
        if (it + 1) % args.grad_accumulation_steps == 0:
            optimizer.update(model, grads)
        # Periodic: report loss, evaluate, save checkpoint
```

### Batch Iteration

```python
def iterate_batches(dataset, batch_size, max_seq_length, train=False):
    # Sorts sequences by length for efficient padding
    # Handles distributed training with average_gradients()
    # Yields (batch_array, lengths_array) tuples
```

### Gradient Checkpointing

```python
def grad_checkpoint(layer):
    # Monkey-patches layer.__call__ to use mx.checkpoint()
    # Trades recomputation for reduced memory during backward pass
    # Applied when TrainingArgs.grad_checkpoint = True
```

### Callbacks

```python
class TrainingCallback:
    # Extensible callback system for training events
    # Override: on_train_loss_report, on_val_loss_report
```

## Dataset System (`tuner/datasets.py`)

### CacheDataset

Pre-tokenized dataset format for efficient training:
```python
class CacheDataset:
    # Loads pre-tokenized sequences from disk
    # Avoids re-tokenization on each epoch
```

### Data Formats

Supported input formats:
- Standard HuggingFace datasets
- JSON / JSONL files
- Parquet files
- SFT (Supervised Fine-Tuning) format with user/assistant roles

### Learning Rate Schedules

```python
def build_schedule(schedule_config):
    # Builds from mlx.optimizers.schedulers
    # Supports warmup via linear_schedule + join_schedules
    # Config: {"name": "cosine_decay", "arguments": [lr, ...], "warmup": steps}
```

## Quantization System (`mlx_lm/quant/`)

### AWQ (`quant/awq.py`)

Activation-Aware Weight Quantization:

```python
@dataclass
class ScaleConfig:
    # Defines which layers to apply scales before/after
    # Per-model configs for Llama, Gemma, DeepSeek, etc.

class AWQConfig:
    # Per-layer quantization configuration
    # Specifies scale computation parameters
```

CLI: `mlx_lm awq --model <model> [--bits 4] [--group-size 128]`

### GPTQ (`quant/gptq.py`)

Post-Training Quantization using gradient information:

CLI: `mlx_lm gptq --model <model> [--bits 4] [--group-size 128]`

### DWQ (`quant/dwq.py`)

Differentiable Weight Quantization -- learns quantization parameters:

CLI: `mlx_lm dwq --model <model>`

### Dynamic Quantization (`quant/dynamic_quant.py`)

Runtime quantization during inference (no calibration data needed):

CLI: `mlx_lm dynamic-quant --model <model>`

### Shared Utilities (`quant/utils.py`)

```python
def load_data(tokenizer, num_samples, max_length):
    # Loads calibration data for quantization methods
    # Used by AWQ, GPTQ, DWQ for calibration passes
```

### Adding LoRA Support for a New Model

1. Verify the model uses standard layer types (`nn.Linear`, etc.)
2. If custom layers exist, implement `to_lora(r, scale, dropout)` method
3. If using `SwitchLinear` (MoE), `LoRASwitchLinear` handles it automatically
4. Test with: `linear_to_lora_layers(model, num_layers=4, config={"rank": 8, "scale": 1.0, "dropout": 0.0})`
5. Verify `print_trainable_parameters()` shows expected percentage
