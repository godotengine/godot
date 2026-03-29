# Workflow Details

!!! info "Scope"
    For artists and integrators who already know which workflow family they need and want the next detailed page.
    This page covers secondary workflow groupings and supporting task pages.
    It complements the canonical [Artist Workflow Overview](../quickstart.md).

## Import Workflow

- Put source files under project `res://`.
- Assign file to `GaussianSplatNode3D`.
- Verify first render before tuning.

References:
- [../../workflows/importing.md](../../workflows/importing.md)
- [../../features/ply-loader.md](../../features/ply-loader.md)

## Tuning Workflow

- Start with balanced preset.
- Adjust render distance and max splat count.
- Validate visual quality and frame stability.

References:
- [../../features/artist_pipeline.md](../../features/artist_pipeline.md)
- [performance-presets.md](performance-presets.md)

## Runtime and Lighting Check Workflow

1. Confirm base runtime stability first (camera movement and visibility).
2. Tune global lighting controls for scene readability.
3. Only then investigate advanced artifacts.

References:
- [runtime-behavior.md](runtime-behavior.md)
- [lighting-behavior.md](lighting-behavior.md)
- [../../troubleshooting/recurring-issues.md](../../troubleshooting/recurring-issues.md)

## Bake / World Workflow

For merged world resources and bake operations:

- [../../workflows/GSPLATWORLD_BAKE.md](../../workflows/GSPLATWORLD_BAKE.md)
