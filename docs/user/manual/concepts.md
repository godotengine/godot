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

- Runtime expectations and first-safe knobs: [runtime-behavior.md](runtime-behavior.md)
- Lighting expectations and first-safe knobs: [lighting-behavior.md](lighting-behavior.md)

## Related

- Manual index: [index.md](index.md)
- Workflows: [workflows.md](workflows.md)
- Performance presets: [performance-presets.md](performance-presets.md)
- FAQ: [faq.md](faq.md)
- Architecture (engineers): [../../architecture/overview.md](../../architecture/overview.md)
