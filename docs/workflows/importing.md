# Gaussian Splat Asset Import Workflow

This is the canonical import page for `.ply` and `.spz` assets.
Use [PLY Loader](../features/ply-loader.md) only when you need the lower-level loader rules.

Visual captures for the import dialog are still pending, so this page stays text-first for now.

## Before You Import

- Keep the source file inside the project so Godot can import it.
- Finish [First Run](../getting-started/quick-start.md) first if you have not confirmed a visible sample yet.
- Choose `.ply` when that is what your source pipeline exports, or `.spz` when you already have a supported compressed asset.

## Supported Paths

| Input | Output | Best for |
| --- | --- | --- |
| `.ply` | Imported `GaussianSplatAsset` resource | Common editor import flow |
| `.spz` | Imported `GaussianSplatAsset` resource | Compressed splat inputs |
| Runtime load path | In-memory `GaussianSplatAsset` | Scripted loading without import-dock output |

## Import Steps

1. Add the source file to the project.
2. Let the editor import it into a `GaussianSplatAsset`.
3. Assign the imported asset to your scene node.
4. Verify a first visible result before you tune quality, lighting, or bake flows.

## What Success Looks Like

| Scenario | Expected result |
| --- | --- |
| Editor import | The import produces a `GaussianSplatAsset` resource you can assign in a scene |
| Runtime load | The load call returns a valid `GaussianSplatAsset` when the file and extension are supported |
| Scene assignment | The node renders once the asset is assigned and the project path is valid |

## Common Failure Modes

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Import fails immediately | The file is not a supported `.ply` or `.spz` asset | Re-export or convert the file into a supported format |
| PLY import fails during validation | Required position, color, scale, rotation, or opacity fields are missing | Re-export with the required property set |
| Imported asset renders with limited shading detail | Optional higher-order SH data is absent | Re-export with the additional SH coefficients if your pipeline supports them |
| SPZ import fails early | The file header or version is unsupported | Recreate the SPZ with a supported toolchain |

## Related Pages

- [Gaussian Splat World Bake Workflow](GSPLATWORLD_BAKE.md)
- [Recurring Issues](../troubleshooting/recurring-issues.md)

## Technical Flow Reference

<figure markdown="1">
![Diagram of the canonical import path from PLY or SPZ sources into GaussianSplatAsset resources](../assets/images/import-workflow-lane.svg){ .gs-diagram }
<figcaption>The import workflow has two entrypoints, but both collapse into the same GaussianSplatAsset type used by scenes, bake scripts, and runtime loads.</figcaption>
</figure>
