# Tests Overview

Use CI runners in `tests/ci/` and runtime harnesses in `tests/runtime/`.

## Core Commands

```bash
python3 tests/ci/run_baseline_qa.py
python3 tests/ci/run_module_tests.py --guard-only
python3 tests/runtime/run_runtime_validation.py
```

## References

- CI runner usage: `tests/ci/README.md`
- Test environment setup: `docs/testing/setup-guide.md`
