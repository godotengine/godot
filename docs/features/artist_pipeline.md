# Gaussian Splat Artist Pipeline

_Last verified: 2026-02-13._

!!! info "Scope"
    For artists and technical artists using inspector tools, hot reload, brush edits, and bake actions inside the editor.
    This page covers in-editor pipeline details after you already know the overall job.
    It complements the canonical [Artist Workflow Overview](../user/quickstart.md).

## Purpose

Use this workflow to import `.ply`/`.spz` assets, iterate with inspector brush tools, and keep node data synchronized with source-file changes.

## Usage

| Task | Action | Implementation reference |
| --- | --- | --- |
| Import a source file | Open the `Gaussian Splatting` bottom panel and click `Import Gaussian`, then select a `res://` `.ply` or `.spz` file. | `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:323`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:329`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:566`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:585` |
| Reimport an existing asset | In `GaussianSplatAsset` inspector, click `Reimport...` to reopen import settings from stored metadata. | `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:320`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1171` |
| Enable runtime debug view (debug builds) | In `Gaussian Splat Overview`, toggle `Runtime Preview` and `Residency HUD`. | `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:753`, `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:760`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:249` |
| Apply brush edits | Set center/radius/strength/hardness/color in `Painterly Brush Tools`, then click `Apply Brush`, `Commit`, or `Revert`. | `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:745`, `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:755`, `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:765`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:944` |
| Bake or restore color grading | Use `Bake Color Grading` or `Restore Original` in the node inspector. | `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:788`, `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:795`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1794`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1821` |

| Hot reload behavior | Current behavior | Implementation reference |
| --- | --- | --- |
| Watch registration | Opening import settings for a source file emits `watch_path_requested` before import confirmation. | `modules/gaussian_splatting/editor/gaussian_import_dialog.cpp:897`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1022` |
| Poll interval | Watches are processed on a 0.1-second timer. | `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1068` |
| `.ply` changes | The editor reimports with stored options and updates the bound node. | `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1125`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1153`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1164` |
| Non-`.ply` watched paths | The resource is reloaded from disk and the node is force-updated. | `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1155`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:1164` |

## API

| API | Behavior | Implementation reference |
| --- | --- | --- |
| `GaussianData.apply_brush_stroke(center, radius, color, opacity, hardness)` | Stages runtime color and opacity edits with radial falloff and records a capped brush history (`2048`). | `modules/gaussian_splatting/core/gaussian_data.cpp:893`, `modules/gaussian_splatting/core/gaussian_data.cpp:907` |
| `GaussianData.commit_runtime_changes()` | Writes staged runtime data into base Gaussian data and clears runtime buffers. | `modules/gaussian_splatting/core/gaussian_data.cpp:814`, `modules/gaussian_splatting/core/gaussian_data.cpp:845` |
| `GaussianData.revert_runtime_changes()` | Discards staged runtime buffers without clearing recorded brush history. | `modules/gaussian_splatting/core/gaussian_data.cpp:848`, `modules/gaussian_splatting/core/gaussian_data.cpp:957` |
| `GaussianData.get_brush_strokes()` | Returns recorded strokes as dictionaries for tooling/serialization hooks. | `modules/gaussian_splatting/core/gaussian_data.cpp:948` |
| `GaussianSplatNode3D.set_runtime_preview_enabled(enabled)` | Switches renderer preview mode to runtime modifications and restores prior mode when disabled. | `modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:682`, `modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:694` |
| `GaussianSplatNode3D.set_show_residency_hud(show)` | Toggles renderer residency HUD and persists the preference into project settings. | `modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:701`, `modules/gaussian_splatting/core/gaussian_splat_settings_manager.cpp:114` |
| `GaussianSplatNode3D.bake_color_grading()` / `restore_color_grading()` | Bakes grading into SH DC colors, disables live grading after bake, and restores original colors on demand. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1794`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1813`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1821` |

## Examples

```gdscript
@tool
extends Node

func apply_and_commit_brush(node: GaussianSplatNode3D) -> void:
    var renderer := node.get_renderer()
    if renderer == null:
        return
    var data := renderer.get_gaussian_data()
    if data == null:
        return
    var local_center := node.to_local(node.global_position)
    data.apply_brush_stroke(local_center, 1.5, Color(1.0, 0.8, 0.6, 1.0), 0.5, 1.0)
    data.commit_runtime_changes()
    node.force_update()
```

```gdscript
@tool
extends Node

func discard_staged_brush_edits(node: GaussianSplatNode3D) -> void:
    var renderer := node.get_renderer()
    if renderer == null:
        return
    var data := renderer.get_gaussian_data()
    if data == null:
        return
    data.revert_runtime_changes()
    node.force_update()
```

## Troubleshooting

| Symptom | Resolution | Implementation reference |
| --- | --- | --- |
| `Runtime Preview` or `Residency HUD` does not appear | Use a debug build because the custom debug inspector controls are compiled only under `DEBUG_ENABLED` and debug properties are hidden in release. | `modules/gaussian_splatting/editor/gaussian_inspector_plugins.cpp:646`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:471` |
| Import dialog rejects the selected file | Move the file under `res://` and retry because the importer rejects non-project paths. | `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:566`, `modules/gaussian_splatting/editor/gaussian_editor_plugin.cpp:577` |
| Brush edits disappear after reimport | Commit edits before reimport because loading new file data resets runtime edits and clears recorded brush strokes. | `modules/gaussian_splatting/core/gaussian_data.cpp:1016`, `modules/gaussian_splatting/core/gaussian_data.cpp:1017` |
| `Bake Color Grading` fails | Assign a `ColorGradingResource` and load Gaussian data before baking. | `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1795`, `modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1800` |
