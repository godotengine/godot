# Color Grading Reference

!!! info "Scope"
    For developers and technical artists who need field ranges, method contracts, and GPU mapping details.
    This page covers exact API lookup after you already know the workflow.
    It complements the canonical [Color Grading Quick Start](../features/color-grading-quick-start.md).

## Purpose
Use this reference for the current color grading API on `GaussianSplatNode3D`, including runtime grading and bake/restore workflows.

## Usage
| Task | Action | Source |
| --- | --- | --- |
| Assign grading to a node | Call `set_color_grading()` or set `rendering/color_grading` with a `ColorGradingResource` | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:190`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:192` |
| Apply grading in real time | Keep resource `enabled=true`; renderer packs values into tile render params each frame | `modules/gaussian_splatting/resources/color_grading_resource.h:10`, `modules/gaussian_splatting/renderer/tile_render_stages.cpp:323` |
| Bake grading into splat data | Call `bake_color_grading()` to update SH DC coefficients in `GaussianData` | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1794`, `modules/gaussian_splatting/core/gaussian_data.cpp:1816` |
| Restore original colors | Call `restore_color_grading()` to restore pre-bake SH DC values | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1821`, `modules/gaussian_splatting/core/gaussian_data.cpp:1855` |

## API
| Resource field | Type | Range/default | Source |
| --- | --- | --- | --- |
| `enabled` | `bool` | default `true` | `modules/gaussian_splatting/resources/color_grading_resource.h:10` |
| `exposure` | `float` | `[-5.0, 5.0]`, default `0.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:11` |
| `contrast` | `float` | `[0.0, 2.0]`, default `1.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:12` |
| `saturation` | `float` | `[0.0, 2.0]`, default `1.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:13` |
| `temperature` | `float` | `[-100.0, 100.0]`, default `0.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:14` |
| `tint` | `float` | `[-100.0, 100.0]`, default `0.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:15` |
| `hue_shift` | `float` | `[-180.0, 180.0]`, default `0.0` | `modules/gaussian_splatting/resources/color_grading_resource.h:16` |

| Node method | Return | Behavior |
| --- | --- | --- |
| `set_color_grading(grading)` | `void` | Assigns resource and marks render state dirty | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1788` |
| `bake_color_grading()` | `Error` | Applies CPU grading to SH DC and disables runtime grading to avoid double-apply | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1794`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1813` |
| `restore_color_grading()` | `void` | Restores backed-up SH DC and re-enables runtime grading | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1821`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1830` |
| `is_color_grading_baked()` | `bool` | Returns bake state from underlying `GaussianData` | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1837` |

| GPU mapping | Layout |
| --- | --- |
| `color_grading_primary` | `x=enabled`, `y=exposure`, `z=contrast`, `w=saturation` (`TileRenderParamsGPU`) |
| `color_grading_secondary` | `x=temperature`, `y=tint`, `z=hue_shift`, `w=reserved` (`TileRenderParamsGPU`) |

Source: `modules/gaussian_splatting/renderer/gaussian_gpu_layout.h:354`, `modules/gaussian_splatting/renderer/tile_render_stages.cpp:323`, `modules/gaussian_splatting/shaders/includes/color_grading_binning.glsl:30`.

## Examples
```gdscript
var grading := ColorGradingResource.new()
grading.enabled = true
grading.exposure = 0.25
grading.contrast = 1.1
grading.saturation = 1.05
grading.temperature = 8.0
grading.tint = -4.0
grading.hue_shift = 6.0

var splat_node := $GaussianSplatNode3D
splat_node.set_color_grading(grading)

var err := splat_node.bake_color_grading()
if err != OK:
    push_error("Bake failed: %s" % err)

if splat_node.is_color_grading_baked():
    splat_node.restore_color_grading()
```

## Troubleshooting
| Symptom | Cause | Action |
| --- | --- | --- |
| `bake_color_grading()` returns `ERR_UNCONFIGURED` | Node has no grading resource or no loaded gaussian data | Assign a `ColorGradingResource` and load data before baking (`modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1795`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1800`) |
| Colors look doubly graded | Runtime grading was re-enabled while baked colors are still applied | Call `restore_color_grading()` before re-baking and keep runtime grading disabled during baked mode (`modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1813`) |
| Runtime sliders do not change output | Resource is unset or disabled | Confirm node property `rendering/color_grading` and `enabled` state (`modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:192`, `modules/gaussian_splatting/resources/color_grading_resource.h:10`) |
