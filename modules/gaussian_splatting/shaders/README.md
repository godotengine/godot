# Gaussian Shader Validation

Canonical validation command (full runtime matrix + contracts):

```bash
python3 modules/gaussian_splatting/shaders/compile_shaders.py \
  --output-dir /tmp/gaussian_shader_validation/spv \
  --summary-json /tmp/gaussian_shader_validation/summary.json
```

Contract-only validation (no SPIR-V compile):

```bash
python3 modules/gaussian_splatting/shaders/compile_shaders.py --contracts-only --list-matrix
```

## Validation Scope

- Runtime stage matrix coverage: `#1267`, `#1318`
- Shader/host ABI contracts: `#1320`
- Per-dispatch counter init contracts: `#1322`
- Diagnostics toggle contracts: `#1324`

## Expected Artifacts

- `summary.json`: machine-readable report with matrix coverage, contract checks, and compile results.
- `spv/*.spv`: one compiled SPIR-V file per matrix entry + variant + shader stage.
