# Compatibility Matrix

Last generated: 2026-02-13

## Purpose
Use this matrix to track validated platform compatibility status for Gaussian Splatting.

## Usage
| Task | Action |
| --- | --- |
| Review current platform status | Read the `Status` and `Notes` columns in the matrix. |
| Update compatibility evidence | Edit `docs/reference/compatibility_sources.yaml` and regenerate this file. |

## API
Entries marked `unknown` have no validated test evidence in this repository.

| Platform | Status | GPU | Driver | Notes |
| --- | --- | --- | --- | --- |
| Windows | unknown | - | - | No validated data in repo; RenderDoc compatibility fallback uses main RenderingDevice only (modules/gaussian_splatting/core/gaussian_splat_manager.cpp:236). |
| Linux | unknown | - | - | No validated data in repo; no checked-in compatibility matrix entries for Linux GPUs. |
| macOS | unknown | - | - | No validated data in repo; no checked-in compatibility matrix entries for Metal/MoltenVK paths. |
| Android | unknown | - | - | No validated data in repo; mobile compatibility requires explicit test evidence. |
| iOS | unknown | - | - | No validated data in repo; mobile compatibility requires explicit test evidence. |

## Examples
```bash
python3 scripts/update_compatibility_matrix.py
```

## Troubleshooting
| Issue | Cause | Fix |
| --- | --- | --- |
| Matrix does not reflect YAML edits | The generator was not rerun. | Run `python3 scripts/update_compatibility_matrix.py`. |
| A platform is still `unknown` | No validated evidence is documented. | Add evidence to `docs/reference/compatibility_sources.yaml` and regenerate. |
| Missing platform row | Platform key was removed from the YAML source. | Re-add the platform entry in `docs/reference/compatibility_sources.yaml`. |
