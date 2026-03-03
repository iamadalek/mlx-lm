# Architecture Overview

Key architectural boundaries, module interactions, and design patterns for mlx-lm.

## Architecture Boundaries

- **Module system with standalone entry points**: Each feature is a standalone
  module with its own `main()` entry point. The CLI router (`mlx_lm/cli.py`)
  dynamically imports submodules by name via `importlib.import_module`. Public
  Python API exported from `mlx_lm/__init__.py`: `load`, `generate`,
  `stream_generate`, `batch_generate`, `convert`.
  (Source: CLAUDE.md "## Architecture > Module Design")

- **Model system**: One file per `model_type` from HuggingFace `config.json` in
  `mlx_lm/models/`. Follows standard Attention -> MLP -> TransformerBlock -> Model
  composition. `MODEL_REMAPPING` in `utils.py` maps aliases to canonical
  implementations (e.g., `"mistral"` -> `"llama"`).
  (Source: CLAUDE.md "## Architecture > Model System")

- **Tuner system**: LoRA/QLoRA fine-tuning in `mlx_lm/tuner/`. `TrainingArgs`
  dataclass for configuration. `linear_to_lora_layers()` in `tuner/utils.py` is
  the extension point for adding LoRA support to new models.
  (Source: CLAUDE.md "## Architecture > Tuner System")

- **Quantization system**: Four methods (AWQ, GPTQ, DWQ, dynamic) in
  `mlx_lm/quant/`, each with own `main()` entry point routed through CLI.
  (Source: CLAUDE.md "## Architecture > Quantization")

- **OpenAI-compatible API server**: `mlx_lm/server.py` provides network-exposed
  inference endpoint. Uses `generate.py` for text generation and `utils.py` for
  model loading.
  (Source: CLAUDE.md "## Architecture > Key Modules")

- **Tool calling system**: Pluggable parsers in `mlx_lm/tool_parsers/` for
  function calling (JSON, Pythonic, Mistral, Devstral, model-specific).
  (Source: CLAUDE.md "## Architecture > Tool Calling")

## Key Abstractions

- **ModelArgs @dataclass pattern**: Every model file uses a `@dataclass ModelArgs`
  for configuration. Also used in `TrainingArgs` for the tuner system.
  (Source: CLAUDE.md "## Code Style")

- **MODEL_REMAPPING dict**: Alias mapping in `utils.py` enables multiple
  HuggingFace `model_type` values to share a single implementation file.
  (Source: CLAUDE.md "## Architecture > Model System")

- **TokenizerWrapper adapter**: Abstraction over HuggingFace tokenizers in
  `tokenizer_utils.py`, providing a unified interface.
  (Source: CLAUDE.md "## Architecture > Key Modules")

- **Standard model composition**: Layered Attention -> MLP -> TransformerBlock ->
  Model structure is the canonical pattern for every model implementation.
  (Source: CLAUDE.md "## Architecture > Model System")
