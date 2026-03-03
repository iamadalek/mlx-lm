# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mlx-lm** is Apple's Python package for running and fine-tuning LLMs on Apple Silicon using the MLX framework. It supports 114+ model architectures, quantization (AWQ/GPTQ/DWQ), LoRA/QLoRA fine-tuning, an OpenAI-compatible API server, and distributed inference. Published to PyPI as `mlx-lm`.

## Development Setup

```bash
pip install -e .                    # Editable install
pip install -e ".[test]"            # With test deps (datasets, lm-eval)
pip install -e ".[train]"           # With training deps (datasets, tqdm)
pip install pre-commit && pre-commit install  # Code formatting hooks
```

## Common Commands

```bash
# Run all tests
python -m unittest discover tests/

# Run a single test file
python -m unittest tests/test_generate.py

# Run a specific test
python -m unittest tests.test_generate.TestGenerate.test_name

# Run distributed tests
mlx.launch -n 2 tests/model_parallel_tests.py

# Format code
pre-commit run --all-files          # black + isort (--profile=black)
pre-commit run --files file1.py     # Single file

# CLI entry points (all accept --help)
mlx_lm generate --model <model>
mlx_lm chat --model <model>
mlx_lm convert --hf-path <model>
mlx_lm lora --model <model> --data <path>
mlx_lm server --model <model>
mlx_lm benchmark --model <model>
```

## Architecture

### Module Design

Each feature is a standalone module with its own `main()` entry point. The CLI router (`cli.py`) dynamically imports submodules by name. Console scripts like `mlx_lm.generate` map directly to `mlx_lm/generate.py:main()`.

### Public Python API

Exported from `mlx_lm/__init__.py`: `load`, `generate`, `stream_generate`, `batch_generate`, `convert`.

### Key Modules

| Module | Purpose |
|--------|---------|
| `generate.py` | Text generation, streaming, batch generation |
| `server.py` | OpenAI-compatible API server |
| `convert.py` | HuggingFace → MLX model conversion with quantization |
| `utils.py` | Model/tokenizer loading, weight handling, `MODEL_REMAPPING` dict |
| `sample_utils.py` | Sampling functions and composable logits processors |
| `cache.py` (models/) | KVCache implementations for all attention variants |
| `tokenizer_utils.py` | TokenizerWrapper abstraction over HF tokenizers |

### Model System (`mlx_lm/models/`)

- Each file corresponds to a `model_type` from HuggingFace `config.json` (e.g., `llama.py` for `"model_type": "llama"`)
- Models use `@dataclass ModelArgs` for configuration and follow a standard structure: `Attention` → `MLP` → `TransformerBlock` → `Model`
- `MODEL_REMAPPING` in `utils.py` maps aliases to canonical implementations (e.g., `"mistral"` → `"llama"`)

### Tuner System (`mlx_lm/tuner/`)

- `trainer.py` — Training loop with `TrainingArgs` dataclass
- `datasets.py` — Dataset loading and preprocessing
- `losses.py` — Loss function implementations
- `utils.py` — LoRA/DoRA layer conversion; edit `linear_to_lora_layers()` to add LoRA support for new models

### Quantization (`mlx_lm/quant/`)

Four methods: AWQ, GPTQ, DWQ, and dynamic quantization. Each has its own module with a `main()` entry point routed through the CLI as `mlx_lm awq`, `mlx_lm gptq`, etc.

### Tool Calling (`mlx_lm/tool_parsers/`)

Pluggable parsers for function calling: JSON, Pythonic, Mistral, Devstral, and model-specific variants.

## Adding a New Model

1. Create `mlx_lm/models/<model_type>.py` matching the `model_type` from the HF `config.json`
2. Implement using the standard pattern (`ModelArgs` dataclass, `Attention`, `MLP`, `TransformerBlock`, `Model` classes)
3. Add LoRA support in `mlx_lm/tuner/utils.py`
4. Add a test in `tests/test_models.py`
5. Determine layer names from the HF `model.safetensors.index.json` or Transformers source

## Testing Notes

- Tests use Python's `unittest` framework (not pytest)
- Some tests require test data downloaded from GitHub releases:
  ```bash
  curl -o test_data.zip -L https://github.com/ml-explore/mlx-lm/releases/download/test_data/test_data.zip
  unzip test_data.zip
  HF_HOME="." python -m unittest discover tests/
  ```
- Platform: Tests require macOS (Darwin) for MLX — will not run on Linux/Windows

## Code Style

- Formatting: `black` + `isort` with black profile (enforced via pre-commit)
- Type hints throughout using `typing` module
- Configuration via `@dataclass` classes (e.g., `ModelArgs`, `TrainingArgs`)
- Copyright header on all files: `# Copyright © 2023-2024 Apple Inc.`
- Version in `mlx_lm/_version.py`, referenced by `setup.py`
