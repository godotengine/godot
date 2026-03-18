# Archived GitHub Actions Workflows

These workflows have been intentionally disabled and moved out of `.github/workflows/` so that GitHub Actions does not execute them. Each file retains a `.disabled` suffix as a reminder that manual action is required to restore the workflow.

## How to Re-enable a Workflow

```bash
# Example: Re-enable the build workflow
mv .github/archived-workflows/build-engine.yml.disabled .github/workflows/build-engine.yml

git add .github/workflows/build-engine.yml
git commit -m "Re-enable build workflow"
```

## Archived Workflows

### `benchmark.yml.disabled`
- **Purpose:** Performance benchmarking on pull requests
- **Triggers:** Pull requests that modify Gaussian Splatting code
- **Status:** Archived until the module implementation stabilises

### `build-engine.yml.disabled`
- **Purpose:** Cross-platform Godot engine builds (Windows, Linux)
- **Triggers:** Pushes and pull requests targeting the `master` and `develop` branches
- **Status:** Archived to avoid resource-intensive builds during early development

Workflows can be revisited once the module reaches a functional state and CI resources are available.
