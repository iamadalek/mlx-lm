# Test Infrastructure Details

CI configuration, test fixture patterns, and platform constraints for mlx-lm.

## Runtime Behaviors Relevant to Testing

- **Batch generation**: `batch_generate` processes multiple prompts simultaneously.
  Tests should verify both single and batched `mx.array` inputs produce correct
  results (observed in `tests/test_sample_utils.py` for `apply_top_p`,
  `apply_min_p`, `apply_top_k`).

## Testing Conventions

- **unittest framework (not pytest)**: All tests use Python's built-in `unittest`
  with `unittest.TestCase` subclasses and `self.assert*` methods.
  (Source: CLAUDE.md "## Testing Notes")

- **macOS-only platform requirement**: Tests require macOS (Darwin) because the
  MLX framework is Apple Silicon-only. Tests will not run on Linux/Windows.
  (Source: CLAUDE.md "## Testing Notes")

- **Test data from GitHub releases**: Some tests require downloading test data:
  `curl -o test_data.zip -L https://github.com/ml-explore/mlx-lm/releases/download/test_data/test_data.zip`
  then setting `HF_HOME="."` before running tests.
  (Source: CLAUDE.md "## Testing Notes")

## Fixture Patterns

- **setUpClass for model loading**: Expensive model loading is done once per
  test class via `@classmethod setUpClass`. Example: `test_generate.py` loads
  `mlx-community/Qwen1.5-0.5B-Chat-4bit` and sets dtype to `float32`.

- **MagicMock for MLX layers**: `unittest.mock.MagicMock` with `spec=` for
  `nn.Linear`, `nn.QuantizedLinear`, `LoRALinear` enables testing parameter
  counting without loading real model weights.

- **setUp/tearDown for stdout**: Capture and restore stdout for testing
  functions that print (e.g., `print_trainable_parameters`).
