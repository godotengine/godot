# Build / Test / CI Reference

This page is the canonical command reference.

## Build

Run from repository root:

=== "Linux"

    ```bash
    scons platform=linuxbsd target=editor dev_build=yes -j$(nproc)
    ```

=== "Windows"

    ```bash
    scons platform=windows target=editor dev_build=yes -j10
    ```

=== "macOS (Apple Silicon)"

    ```bash
    scons platform=macos target=editor dev_build=yes arch=arm64 -j8
    ```

For test-enabled editor builds:

```bash
scons platform=<platform> target=editor dev_build=yes tests=yes -j<jobs>
```

> **Binary naming:** `dev_build=yes` adds a `.dev` segment to the output binary name.
> For example, Windows produces `bin/godot.windows.editor.dev.x86_64.exe`
> (not `bin/godot.windows.editor.x86_64.exe`).

## Test Runners

- Baseline QA:
  - `python3 tests/ci/run_baseline_qa.py --godot <module-built-binary>`
- Module checks/tests:
  - `python3 tests/ci/run_module_tests.py --guard-only`
  - `python3 tests/ci/run_module_tests.py --godot-binary <module-built-binary>`
- Runtime validation:
  - `python3 tests/runtime/run_runtime_validation.py --godot-binary <module-built-binary> --gd-mode headless`
- Benchmark suite:
  - `python3 tests/runtime/run_benchmark.py --godot-binary <module-built-binary> --profile everything`

## CI Source of Truth

- Workflow overview: [../../.github/workflows/README.md](../../.github/workflows/README.md)
- Production gate workflow: [../../.github/workflows/gaussian_production_gates.yml](../../.github/workflows/gaussian_production_gates.yml)

## Common Failure Modes

- Wrong binary (stock Godot instead of module-enabled build)
- Missing toolchain dependencies (`scons`, shader compiler toolchain)
- Build path mismatch (for example using a stale editor outside this fork's `bin/` output)

Use recurring fixes:

- [../troubleshooting/recurring-issues.md](../troubleshooting/recurring-issues.md)
