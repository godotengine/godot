# GitHub Actions Workflows

This directory contains 6 active workflows.

## Active Workflows

| Workflow | File | Purpose | Notes |
| --- | --- | --- | --- |
| Baseline QA Automation | `baseline_qa.yml` | Runs baseline QA and optional compiled-module QA. | Builds module-enabled Godot (``); nightly schedule runs update mode and refreshes `qa-regression-baseline`. |
| Gaussian Pipeline Validation | `gaussian_pipeline_validation.yml` | Builds module and runs pipeline smoke tests. | Broad push/PR coverage. |
| Gaussian Production Gates | `gaussian_production_gates.yml` | Enforces guard checks, Linux module validation, and optional Windows GPU evidence lane. | Canonical production gate workflow. |
| Gaussian Shader Validation | `gaussian_shader_validation.yml` | Validates shader compile matrix and host/shader contract checks. | Focused shader CI gate. |
| Gaussian Splatting Tests | `test_gaussian_splatting.yml` | Multi-tier suite (docker, compiled, external-tool compatibility). | Includes summary/artifact aggregation. |
| Phase 4C Test Suite | `test_phase4.yml` | Legacy-named cross-platform suite (Windows/Linux). | Manual-only (`workflow_dispatch`) for historical compatibility checks. |

## Manual Dispatch Inputs

| Workflow | Input | Options |
| --- | --- | --- |
| `baseline_qa.yml` | `debug_mode` | `true`, `false` |
| `baseline_qa.yml` | `baseline_mode` | `compare`, `update` |
| `gaussian_production_gates.yml` | `run_gpu_lane` | `true`, `false` |
| `gaussian_production_gates.yml` | `enforce_gpu_readiness` | `true`, `false` |
| `gaussian_production_gates.yml` | `runtime_loops` | integer string |
| `test_gaussian_splatting.yml` | `test_category` | `all`, `ply`, `pipeline`, `sorting`, `runtime`, `module`, `qa` |
| `test_phase4.yml` | `test_category` | `all`, `ply`, `pipeline`, `sorting`, `runtime`, `module`, `qa` |

## Scheduled Triggers

| Workflow | Schedule (UTC) | Behavior |
| --- | --- | --- |
| `baseline_qa.yml` | `30 3 * * *` | Runs in update mode and publishes `qa-regression-baseline` for future compare runs. |

## Dependencies

- Python 3.11
- SCons/build toolchain for compiled lanes
- Vulkan-capable environment for render-path lanes
- `xvfb` for Linux non-headless rendering checks

## Archived Workflows

Disabled workflows are stored in `../archived-workflows/`.
