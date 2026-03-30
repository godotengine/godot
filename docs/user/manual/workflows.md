# Workflow Details

Use this page after [Artist Workflow Overview](../quickstart.md) when you already know the job and need the next supporting page.

## Import Workflow

- Put source files inside the project.
- Assign the imported asset to `GaussianSplatNode3D`.
- Verify a first render before you tune quality or lighting.

Related pages:
- [Import workflow](../../workflows/importing.md)
- [PLY loader reference](../../features/ply-loader.md)

## Tuning Workflow

- Start with `Balanced`.
- Adjust render distance and max splat count only after the scene is visible.
- Validate image quality and frame stability before going deeper.

Related pages:
- [Artist pipeline reference](../../features/artist_pipeline.md)
- [Performance presets](performance-presets.md)

## Runtime and Lighting Checks

1. Confirm base runtime stability first.
2. Tune the lighting controls that change readability the most.
3. Use troubleshooting only after the normal controls fail.

Related pages:
- [Runtime behavior](runtime-behavior.md)
- [Lighting behavior](lighting-behavior.md)
- [Recurring issues](../../troubleshooting/recurring-issues.md)

## Bake and World Workflows

Use the bake workflow when you want one merged runtime world resource instead of several source assets.

- [Gaussian Splat World Bake Workflow](../../workflows/GSPLATWORLD_BAKE.md)
