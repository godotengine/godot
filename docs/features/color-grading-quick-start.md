# Color Grading Quick Start

Last updated: 2026-02-13

!!! tip "Canonical task page"
    Use this page as the source of truth for applying, tuning, baking, and restoring color grading in a scene workflow.
    For exact fields and API lookup, use [Color Grading Reference](../reference/color-grading.md) as the technical complement.

## Purpose

Use `ColorGradingResource` on `GaussianSplatNode3D` for real-time grading and optional baking into SH DC color data.

## Usage

| Step | Action | Implementation reference |
| --- | --- | --- |
| 1 | Assign a `ColorGradingResource` to `rendering/color_grading` on the node. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:192` |
| 2 | Tune `exposure`, `contrast`, `saturation`, `temperature`, `tint`, and `hue_shift` on the resource. | `modules/gaussian_splatting/resources/color_grading_resource.cpp:20` |
| 3 | Keep `enabled = true` for real-time grading in tile binning. | `modules/gaussian_splatting/resources/color_grading_resource.cpp:15`, `modules/gaussian_splatting/shaders/includes/color_grading_binning.glsl:32` |
| 4 | Run `bake_color_grading()` or click `Bake Color Grading` to write grading into base SH DC values. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1794`, `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:788` |
| 5 | Run `restore_color_grading()` or click `Restore Original` to restore pre-bake colors. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1821`, `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:795` |

## API

| Item | Type | Behavior | Implementation reference |
| --- | --- | --- | --- |
| `rendering/color_grading` | Node property | Stores the grading resource used by the renderer. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:192` |
| `set_color_grading(grading)` | Method | Updates node state and marks render state dirty for re-upload. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1788` |
| `bake_color_grading()` | Method | Applies grading on CPU to SH DC coefficients and disables grading to avoid double-application. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1794`, `modules/gaussian_splatting/core/gaussian_data.cpp:1816` |
| `restore_color_grading()` | Method | Restores original SH DC coefficients and re-enables grading. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1821`, `modules/gaussian_splatting/core/gaussian_data.cpp:1855` |
| `is_color_grading_baked()` | Method | Reports whether bake state is active in `GaussianData`. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1837`, `modules/gaussian_splatting/core/gaussian_data.h:763` |

| `ColorGradingResource` field | Default | Clamp range | Implementation reference |
| --- | --- | --- | --- |
| `enabled` | `true` | bool | `modules/gaussian_splatting/resources/color_grading_resource.h:10` |
| `exposure` | `0.0` | `-5.0..5.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:11` |
| `contrast` | `1.0` | `0.0..2.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:12` |
| `saturation` | `1.0` | `0.0..2.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:13` |
| `temperature` | `0.0` | `-100.0..100.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:14` |
| `tint` | `0.0` | `-100.0..100.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:15` |
| `hue_shift` | `0.0` | `-180.0..180.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:16` |

| Pipeline step | Behavior | Implementation reference |
| --- | --- | --- |
| Resource registration | `ColorGradingResource` is already registered at module init. | `modules/gaussian_splatting/register_types.cpp:94` |
| Build inclusion | `resources/*.cpp` is already compiled by module `SCsub`. | `modules/gaussian_splatting/SCsub:33`, `modules/gaussian_splatting/SCsub:47` |
| Renderer upload | Renderer writes grading values into render params each frame. | `modules/gaussian_splatting/renderer/tile_render_stages.cpp:321` |
| GPU layout | Render params include two `vec4` grading fields. | `modules/gaussian_splatting/renderer/gaussian_gpu_layout.h:353`, `modules/gaussian_splatting/shaders/includes/gs_render_params.glsl:58` |
| Shader application | Tile binning applies grading after SH evaluation, including cached SH path. | `modules/gaussian_splatting/shaders/tile_binning.glsl:1358`, `modules/gaussian_splatting/shaders/tile_binning.glsl:1365` |

## Examples

```gdscript
extends Node3D

@onready var splat: GaussianSplatNode3D = $GaussianSplatNode3D

func _ready() -> void:
	var grading := ColorGradingResource.new()
	grading.enabled = true
	grading.exposure = 0.35
	grading.contrast = 1.1
	grading.saturation = 1.15
	grading.temperature = 8.0
	grading.tint = -4.0
	grading.hue_shift = 6.0
	splat.set_color_grading(grading)

func bake_grading() -> void:
	var err := splat.bake_color_grading()
	if err != OK:
		push_warning("bake_color_grading failed: %d" % err)

func restore_grading() -> void:
	splat.restore_color_grading()
```

## Troubleshooting

| Symptom | Cause | Fix | Implementation reference |
| --- | --- | --- | --- |
| `bake_color_grading()` returns an error | No grading resource or no loaded gaussian data. | Assign `rendering/color_grading` and load data before baking. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1795`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1800` |
| Inspector changes have no visible effect | Grading is disabled or resource is not assigned. | Set `enabled = true` and re-check the node property assignment. | `modules/gaussian_splatting/resources/color_grading_resource.cpp:15`, `modules/gaussian_splatting/renderer/tile_render_stages.cpp:322` |
| Colors look graded twice after manual workflow changes | Grading remained enabled after custom bake flow. | Disable grading after bake or call node bake API which already disables it. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1811` |
| Restore does not bring back expected base colors | No previous bake state was recorded. | Bake once before expecting restore behavior. | `modules/gaussian_splatting/core/gaussian_data.cpp:1822`, `modules/gaussian_splatting/core/gaussian_data.cpp:1857` |
