# Reviewer Fast Path

!!! tip "Use this page when you need the shortest review loop"
    Start here if you want to understand what godotGS changes, where the fork delta lives, and what to validate first.

This path is for reviewers, maintainers, and contributors who want to understand the fork without rebuilding the whole repository first.

## Read First

- [Repository README](https://github.com/klausi3D/godotGS/blob/master/README.md) for the public front door and current project status.
- [Engine Patches](https://github.com/klausi3D/godotGS/blob/master/ENGINE_PATCHES.md) for the engine-level delta.
- [Architecture Overview](../architecture/overview.md) for subsystem boundaries.
- [Compatibility Matrix](../reference/compatibility-matrix.md) for current evidence-backed platform status.
- [Build / Test / CI Command Reference](../reference/build-test-ci.md) for the validation entrypoints.

## What To Check

1. Confirm the project is still Alpha and that the preferred public evaluation path is Linux nightly.
2. Read `ENGINE_PATCHES.md` before opening engine-root changes.
3. Inspect `modules/gaussian_splatting/` for the module implementation that owns the fork delta.
4. Use the sample project in `tests/examples/godot/test_project` to verify the current runtime path.
5. Check the compatibility matrix before assuming a platform claim is production-ready.

## Validation Shortlist

Use these commands when you want to sanity-check the public docs and reviewer surface:

```bash
python scripts/docs/check_links.py docs README.md BUILDING.md CONTRIBUTING.md
```

```bash
python scripts/docs/release_acceptance.py
```

## If You Need To Build

- [Build from Source](../BUILDING.md) for a module-enabled editor from this fork.
- [First Run](../getting-started/quick-start.md) for the sample-project path after the editor is built.
- [Try in 5 Minutes](../getting-started/try-in-5-minutes.md) for the fastest nightly-first evaluation loop.
