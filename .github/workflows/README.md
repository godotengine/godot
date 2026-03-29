# GitHub Actions Workflows

This directory contains 5 active workflow files.

GitHub's Actions tab can also show historical workflow names from past runs, disabled files, or workflow files that are no longer present in this directory. This README tracks the workflow files currently checked into `.github/workflows/`.

## Active Workflows

| Workflow | File | Purpose | Notes |
| --- | --- | --- | --- |
| Baseline QA Automation | `baseline_qa.yml` | Runs baseline QA and optional compiled-module QA. | Builds the Linux editor once and reuses that artifact for push-only compiled QA. |
| Docs Pages (Versioned) | `docs_pages.yml` | Builds and deploys MkDocs docs with mike versioning to `gh-pages`. | Publishes `latest` from `master/main` and versioned docs from `v*` tags. |
| Gaussian Production Gates | `gaussian_production_gates.yml` | Enforces guard checks, pipeline smoke, runtime validation, benchmark probes, and optional Windows GPU evidence lane. | Owns the single Windows build for validation workflows. |
| Gaussian Shader Validation | `gaussian_shader_validation.yml` | Validates shader compile matrix and host/shader contract checks. | Focused shader CI gate. |
| Release Builds | `release_builds.yml` | Builds the Linux editor for CI artifacts, nightly prereleases, and optional stable-tag publishes. | Current public binary coverage is Linux editor only. |

## Manual Dispatch Inputs

| Workflow | Input | Options |
| --- | --- | --- |
| `baseline_qa.yml` | `debug_mode` | `true`, `false` |
| `baseline_qa.yml` | `baseline_mode` | `compare`, `update` |
| `gaussian_production_gates.yml` | `run_gpu_lane` | `true`, `false` |
| `gaussian_production_gates.yml` | `enforce_gpu_readiness` | `true`, `false` |
| `gaussian_production_gates.yml` | `runtime_loops` | integer string |
| `release_builds.yml` | `publish_channel` | `none`, `nightly`, `stable` |
| `release_builds.yml` | `release_tag` | string (`vX.Y.Z` when `publish_channel=stable`) |
| `release_builds.yml` | `release_name` | optional string |
| `release_builds.yml` | `keep_nightlies` | integer string |

## Scheduled Triggers

| Workflow | Schedule (UTC) | Behavior |
| --- | --- | --- |
| `baseline_qa.yml` | `30 3 * * *` | Runs in update mode and publishes `qa-regression-baseline` for future compare runs. |
| `release_builds.yml` | `30 2 * * *` | Builds and publishes the nightly prerelease, then prunes older nightly releases and tags. |

## Dependencies

- Python 3.11
- SCons/build toolchain for compiled lanes
- Self-hosted Windows runner attached to this repository with labels `self-hosted`, `Windows`, `X64`, `godotgs`
- Optional GPU evidence label `gpu` for the Windows evidence lane
- Vulkan-capable environment for render-path lanes
- `xvfb` for Linux non-headless rendering checks

## Archived Workflows

Disabled workflows are stored in `../archived-workflows/`.

- `benchmark.yml.disabled`
- `build-engine.yml.disabled`
- `gaussian_pipeline_validation.yml.disabled`
- `test_gaussian_splatting.yml.disabled`
- `test_phase4.yml.disabled`
