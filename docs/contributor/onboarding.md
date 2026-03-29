# Contributor Onboarding

Goal: get a local build, run canonical checks, and ship a safe first PR.

## 1) Local Setup

- Build from source: [../BUILDING.md](../BUILDING.md)
- Test and CI command reference: [../reference/build-test-ci.md](../reference/build-test-ci.md)

## 2) Run Canonical Validation

- Guard-only checks:
  - `python3 tests/ci/run_module_tests.py --guard-only`
- Baseline QA:
  - `python3 tests/ci/run_baseline_qa.py --godot <module-built-binary>`
- Runtime validation:
  - `python3 tests/runtime/run_runtime_validation.py --godot-binary <module-built-binary> --gd-mode headless`

## 3) Understand the Codebase Quickly

- Module overview: [../../modules/gaussian_splatting/README.md](../../modules/gaussian_splatting/README.md)
- Architecture overview: [../architecture/overview.md](../architecture/overview.md)
- Render pipeline details: [../architecture/render-pipeline.md](../architecture/render-pipeline.md)
- Lighting details: [../architecture/lighting-system.md](../architecture/lighting-system.md)
- API index: [../api/index.md](../api/index.md)

## 4) First PR Expectations

- Keep scope narrow.
- Include test evidence.
- Update docs when user-visible behavior changes.
- Follow standards in [../governance/contribution-standards.md](../governance/contribution-standards.md).
