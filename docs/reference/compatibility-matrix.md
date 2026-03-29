# Compatibility Matrix

Last generated: 2026-03-29

## Purpose
Use this matrix to track evidence-backed platform status for godotGS Gaussian Splatting.

## Usage
| Task | Action |
| --- | --- |
| Review current platform status | Read the `Current state`, `Public binaries`, and `Notes` columns in the platform table. |
| Update compatibility evidence | Edit `docs/reference/compatibility_sources.yaml` and regenerate this file. |

<figure markdown="1">
![Diagram of the compatibility evidence ladder with current platform positions](../assets/images/compatibility-evidence-ladder.svg){ .gs-diagram }
<figcaption>The matrix is a ladder, not a badge wall: each platform only claims the highest evidence state the repository can currently prove.</figcaption>
</figure>

## Evidence Levels
| Level | Meaning |
| --- | --- |
| `unsupported` | The build system rejects the platform. |
| `build-supported` | The build system accepts the platform and the repository can compile it. |
| `smoke-tested` | A minimal runtime, import, or QA lane passes, but not necessarily a non-headless editor lane. |
| `editor-tested` | A non-headless editor or runtime validation lane passes in repo-owned automation. |
| `sample-project-tested` | A published example-project or QA-scene lane passes on the platform. |
| `production-tested` | Published release-signoff or field evidence exists for the platform. |

## Platform States
The state shown for each platform is the strongest evidence level currently documented in this repository.

| Platform | Current state | Public binaries | Evidence | Notes |
| --- | --- | --- | --- | --- |
| Windows | `editor-tested` | No public Windows binaries at present; build from source. | SUPPORTED_PLATFORMS in modules/gaussian_splatting/config.py accepts windows.<br>Gaussian Production Gates builds a Windows editor and runs windows-vulkan runtime validation in .github/workflows/gaussian_production_gates.yml.<br>The production evidence route in tests/ci/collect_production_evidence.ps1 exercises non-headless Vulkan runtime scripts on Windows. | Windows is the strongest in-repo validation lane currently, but the public binary surface is still narrower than the source-support story. |
| Linux | `smoke-tested` | Primary public binary lane: Linux editor CI artifacts and nightly prereleases. | SUPPORTED_PLATFORMS in modules/gaussian_splatting/config.py accepts linuxbsd.<br>Release Builds compiles and publishes the Linux editor in .github/workflows/release_builds.yml.<br>Baseline QA builds a Linux editor, imports the project, and runs baseline QA under xvfb in .github/workflows/baseline_qa.yml. | Linux is currently the only platform with public binaries. The documented validation is still mostly CI and headless-oriented, so it stays below editor-tested. |
| macOS | `build-supported` | No public macOS binaries at present; build from source. | SUPPORTED_PLATFORMS in modules/gaussian_splatting/config.py accepts macos. | The repo allows macOS builds, but no published smoke, editor, or sample-project evidence is checked in yet. |
| Android | `unsupported` | None. | Android is not listed in SUPPORTED_PLATFORMS in modules/gaussian_splatting/config.py. | The build system rejects this platform. |
| iOS | `unsupported` | None. | iOS is not listed in SUPPORTED_PLATFORMS in modules/gaussian_splatting/config.py. | The build system rejects this platform. |

## Published Test Environments
These rows record the concrete hardware/software environments that are currently described in repo-owned automation or published evidence. Missing adapter or driver names mean the lane exists, but those identifiers are not yet published in the docs.

| Platform | State | Hardware | Software | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Windows | `editor-tested` | Self-hosted Windows GPU runner (adapter name and driver version not yet published) | Windows display driver + Vulkan renderer + Forward+ + safe render thread | .github/workflows/gaussian_production_gates.yml module-validation lane<br>tests/runtime/run_runtime_validation.py --gd-mode windows-vulkan<br>tests/ci/collect_production_evidence.ps1 | Concrete GPU/driver identifiers are still missing from the public docs, so this row describes the validated software lane without claiming a specific adapter. |
| Linux | `smoke-tested` | GitHub-hosted Ubuntu runner (GPU and driver version not yet published) | ubuntu-latest + xvfb + mesa-vulkan-drivers + headless import + baseline QA | .github/workflows/baseline_qa.yml CPU-only lane<br>.github/workflows/release_builds.yml Linux editor build lane | This is the current public binary platform and the best documented CI lane, but it is still not an interactive editor validation row. |

## Reading the Matrix
- `build-supported` means the build system accepts the platform and repo automation can compile it.
- `smoke-tested` means a minimal runtime or QA lane has passed, but not necessarily an interactive editor lane.
- `editor-tested` means a non-headless editor/runtime lane has passed.
- `sample-project-tested` and `production-tested` are reserved for stronger published evidence than this repo currently exposes publicly.

## Examples
```bash
python3 scripts/update_compatibility_matrix.py
```

## Troubleshooting
| Issue | Cause | Fix |
| --- | --- | --- |
| Matrix does not reflect YAML edits | The generator was not rerun. | Run `python3 scripts/update_compatibility_matrix.py`. |
| A platform only reaches `build-supported` | The repo has no documented runtime or editor evidence for it yet. | Add stronger evidence to `docs/reference/compatibility_sources.yaml` only after the lane or test result exists. |
| Hardware or driver fields are still generic | The lane exists, but the adapter or driver identifiers are not yet published in repo-owned evidence. | Capture them from the evidence run and replace the placeholder text in `docs/reference/compatibility_sources.yaml`. |
| Missing platform row | Platform key was removed from the YAML source. | Re-add the platform entry in `docs/reference/compatibility_sources.yaml`. |
