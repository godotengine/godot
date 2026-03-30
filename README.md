# Godot Gaussian Splatting (Alpha)

GodotGS is a Godot 4.5 fork with an in-tree Gaussian Splatting module for importing, rendering, and tuning splat-based scenes in the editor and at runtime.

## Current Status

| Area | State |
| --- | --- |
| Maturity | Alpha |
| Fastest public evaluation path | Linux nightly editor |
| Windows / macOS | Source build first |
| Public binaries | Linux editor only |
| Compatibility truth | [Compatibility Matrix](docs/reference/compatibility-matrix.md) |
| Performance truth | [Performance Dashboard](docs/performance/index.md) |

No named non-nightly release is published yet. If you want the fastest way to evaluate the project, use the Linux nightly path first.

## Who This Is For

- Technical artists and graphics engineers evaluating Gaussian Splatting inside a Godot 4.5 fork
- Contributors who need an in-tree module plus engine-patch context, not a standalone plugin
- Reviewers who want to separate the upstream Godot tree from the godotGS-specific delta quickly

## Fastest Way In

1. [Try in 5 Minutes](docs/getting-started/try-in-5-minutes.md) if you want the shortest honest evaluation path.
2. [First Run](docs/getting-started/quick-start.md) if you want the canonical sample-project flow.
3. [Compatibility Matrix](docs/reference/compatibility-matrix.md) if you need platform evidence before trying it.
4. [Build from Source](docs/BUILDING.md) if you are on Windows or macOS, or you want a custom editor binary.

## Current Public Evidence

- Compatibility snapshot: Windows is `editor-tested` on the self-hosted Vulkan Forward+ lane with `NVIDIA GeForce RTX 3090`; Linux is `sample-project-tested` on `ubuntu-24.04` with `xvfb` and `mesa-vulkan-drivers 25.2.8-0ubuntu0.24.04.1`; macOS is currently `build-supported`.
- Benchmark snapshot: the public dashboard currently contains one committed `static_baseline` row at 74.0 average FPS and 15.62 ms P99 frame time.
- Visual proof: real editor screenshots and short workflow clips are still pending. The current figures are technical diagrams, not product captures.

## For Reviewers

- [Reviewer Fast Path](docs/contributor/reviewer-fast-path.md)
- [Engine Patches](ENGINE_PATCHES.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Build / Test / CI Command Reference](docs/reference/build-test-ci.md)

## Documentation

- [Documentation home](docs/index.md)
- [Public roadmap](https://github.com/klausi3D/godotGS/issues/186)
- [Docs site maintenance guide](docs/development/docs-site.md)
- [Contribute](docs/contributor/index.md)
- [User Guide](docs/user/index.md)
- [Reference](docs/reference/index.md)

## Repository Layout

- [Engine root](./): upstream Godot now lives at repository root.
- [Gaussian Splatting module](modules/gaussian_splatting/): module implementation.
- [Test harnesses](tests/): CI and runtime validation tooling.
- [Documentation](docs/): user, contributor, architecture, and reference docs.

## License

Repository code and documentation are MIT-licensed unless noted otherwise.
Upstream engine code at repository root follows upstream licensing.
