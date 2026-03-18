# Synthetic Splat Baseline Artifacts

These JSON files are deterministic example outputs for the synthetic splat generators used by the Gaussian Splatting test suite.

## Regenerate

```bash
python3 modules/gaussian_splatting/tests/generate_synthetic_splat_baselines.py
```

## Verify in QA

```bash
python3 modules/gaussian_splatting/tests/generate_synthetic_splat_baselines.py --check
python3 modules/gaussian_splatting/tests/run_phase1_tests.py --godot-path . --project-path . --test-only synthetic
```

## Determinism contract

- Seeded SplitMix64 (`seed`) drives all sampled values.
- Config hashing (`config_hash`) uses FNV-1a with float quantization (`100000.0` scale).
- Scene hashing (`scene_hash`) includes seed/config hash and all generated splat summary fields.
- Any seed/config change should change at least one hash.
