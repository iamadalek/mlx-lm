---
name: testing-patterns
description: >
  Python unittest conventions for mlx-lm: setUpClass shared model fixtures,
  MagicMock-based MLX layer mocking, batch-mode test patterns for MLX array
  operations, and macOS-only platform requirements.
domains: [testing, quality, coverage]
user-invocable: false
generated_by: structured-workflows:setup
generated_at: "2026-03-02"
---

## Detected in This Project

- `tests/test_generate.py`
- `tests/test_tuner_utils.py`
- `tests/test_sample_utils.py`
- `tests/test_losses.py`
- `tests/test_tokenizers.py`
- `tests/test_tool_parsing.py`
- `tests/test_tuner_trainer.py`
- `tests/test_utils.py`
- `tests/test_datsets.py`
- `tests/test_models.py`

## Testing Conventions

### Naming

This project uses `test_{module}.py` files in a dedicated `tests/` directory,
mirroring the source module structure. Each test file maps to a source module
in `mlx_lm/` (e.g., `test_generate.py` tests `generate.py`,
`test_tuner_utils.py` tests `tuner/utils.py`).

### Framework

Python's built-in `unittest` framework is used exclusively — not pytest.
Tests subclass `unittest.TestCase` and use `self.assert*` methods:
- `self.assertEqual`, `self.assertTrue`, `self.assertAlmostEqual`
- `self.assertRaises` for expected exceptions
- Observed in `tests/test_generate.py`, `tests/test_sample_utils.py`,
  `tests/test_tuner_utils.py`

### Structure

- **Shared expensive fixtures**: `setUpClass` loads models once per test class
  (observed in `tests/test_generate.py` — loads `Qwen1.5-0.5B-Chat-4bit` and
  sets dtype to `float32`)
- **Per-test state management**: `setUp`/`tearDown` for stdout capture and
  reset (observed in `tests/test_tuner_utils.py`)
- **MagicMock for MLX layers**: `unittest.mock.MagicMock` with `spec=nn.Linear`,
  `spec=nn.QuantizedLinear`, `spec=LoRALinear` for testing parameter counting
  without real model weights (observed in `tests/test_tuner_utils.py`)
- **Batch-mode verification**: Tests verify functions work for both single and
  batched `mx.array` inputs (observed in `tests/test_sample_utils.py` —
  `apply_top_p`, `apply_min_p`, `apply_top_k` each have batch-mode assertions)
- One test class per file, flat method structure

### Tooling

- Test runner: `python -m unittest discover tests/`
- Platform: macOS (Darwin) required — MLX framework does not run on Linux/Windows
- Test data: some tests require data downloaded from GitHub releases
- Optional test deps via `pip install -e ".[test]"`: `datasets`, `lm-eval`
- Code formatting: `pre-commit` with `black` + `isort` (enforced on all files)

See `references/test-infrastructure.md` for CI configuration, test fixture patterns, and platform constraints.
