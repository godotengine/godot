# Compatibility Matrix

Last generated: 2026-03-19

## Purpose
Use this matrix to track validated platform compatibility status for Gaussian Splatting.

## Usage
| Task | Action |
| --- | --- |
| Review current platform status | Read the `Status` and `Notes` columns in the matrix. |
| Update compatibility evidence | Edit `docs/reference/compatibility_sources.yaml` and regenerate this file. |

## Platform Support

Status is derived from `SUPPORTED_PLATFORMS` in `modules/gaussian_splatting/config.py`, which
gates the build via `can_build()`. Platforms not in that set are rejected at compile time.

| Platform | Status | GPU | Driver | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Windows | supported | - | - | `SUPPORTED_PLATFORMS` in `config.py` | Primary development platform. RenderDoc compatibility fallback uses main RenderingDevice only. |
| Linux | supported | - | - | `SUPPORTED_PLATFORMS` (as `linuxbsd`) in `config.py` | Builds and runs on desktop Linux distributions. |
| macOS | supported | - | - | `SUPPORTED_PLATFORMS` in `config.py` | Builds via Metal/MoltenVK path. |
| Android | unsupported | - | - | Not in `SUPPORTED_PLATFORMS` in `config.py` | Build system rejects this platform. |
| iOS | unsupported | - | - | Not in `SUPPORTED_PLATFORMS` in `config.py` | Build system rejects this platform. |

**Note:** "supported" means the build system accepts the platform and compilation succeeds.
Per-GPU and per-driver validation data has not yet been collected; those columns will be
populated as hardware-specific test results become available.

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
