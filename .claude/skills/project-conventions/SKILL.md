---
name: project-conventions
description: >
  Project structure and conventions for mlx-lm: dynamic CLI routing via
  importlib, @dataclass ModelArgs/TrainingArgs configuration pattern,
  setup.py-based packaging with 18 console script entry points, and
  MLX/HuggingFace dependency management.
domains: [conventions, structure, patterns]
user-invocable: false
generated_by: structured-workflows:setup
generated_at: "2026-03-02"
---

## Detected in This Project

- `setup.py`
- `.pre-commit-config.yaml`

## Project Conventions

### Module Organization

This project uses a dynamic CLI router pattern. `mlx_lm/cli.py` dynamically
imports submodules via `importlib.import_module(f"mlx_lm.{subcommand}")` and
calls their `main()` entry point. Quantization subcommands (`awq`, `dwq`,
`dynamic_quant`, `gptq`) are mapped through a `subpackages` dict to
`quant.{subcommand}`. Observed in `mlx_lm/cli.py`.

The public Python API is explicitly exported from `mlx_lm/__init__.py`:
`convert`, `batch_generate`, `generate`, `stream_generate`, `load` — with
`__all__` to control the public surface. The `TRANSFORMERS_NO_ADVISORY_WARNINGS`
env var is set globally at import time. Observed in `mlx_lm/__init__.py`.

### Build and Tooling

- **Packaging**: `setup.py` with `setuptools` (not pyproject.toml)
- **Entry points**: 18 console scripts mapping `mlx_lm.{command}` to
  `mlx_lm/{module}:main()` functions
- **Formatting**: `pre-commit` hooks with `black` (v25.1.0) and `isort`
  (`--profile=black`, v6.0.0)
- **Copyright**: All files carry `# Copyright © 2023-2024 Apple Inc.` header
- **Type hints**: Used throughout via the `typing` module
- **Configuration**: `@dataclass` classes for all configuration (e.g.,
  `ModelArgs`, `TrainingArgs`)

### Dependency Management

- **Core deps**: `mlx` (platform-conditional, Darwin only), `numpy`,
  `transformers>=5.0.0`, `sentencepiece`, `protobuf`, `pyyaml`, `jinja2`
- **Version pinning**: `MIN_MLX_VERSION = "0.30.4"` used for mlx floor;
  transformers pinned to `>=5.0.0`
- **Optional extras**: `test` (datasets, lm-eval), `train` (datasets, tqdm),
  `evaluate` (lm-eval, tqdm), `cuda13`/`cuda12`/`cpu` for MLX backend variants
- **Python**: `>=3.8`

See `references/architecture-overview.md` for architectural boundaries, module interactions, and design patterns.
