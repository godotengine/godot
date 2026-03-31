# User Manual: Concepts

## What This Module Does

Gaussian splatting renders scenes from splat data (`.ply` / `.spz`) instead of traditional meshes.

## Core Concepts (Plain Language)

- Splat asset: source point/splat data file you import.
- Node: `GaussianSplatNode3D` is the scene node that renders the asset.
- Quality preset: quick performance/quality control.
- Streaming: loads scene data in chunks to keep memory under control.
- Sorting: draw order step needed for correct translucent rendering.

## When to Tune

- Use presets first.
- Only change advanced settings when you see visible artifacts or performance problems.

## Behavior Guides

- [Runtime behavior](runtime-behavior.md) for expected behavior and the first safe controls.
- [Lighting behavior](lighting-behavior.md) for the lighting controls that matter first.

## Related

- [User manual home](index.md)
- [Workflow details](workflows.md)
- [Performance presets](performance-presets.md)
- [FAQ](faq.md)
- [Architecture overview](../../architecture/overview.md) if you need the engine-level model.
