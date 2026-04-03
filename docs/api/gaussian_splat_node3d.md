# GaussianSplatNode3D API Reference

## Purpose
Use `GaussianSplatNode3D` to render Gaussian splat assets or procedural splat arrays in a `Node3D` scene (`modules/gaussian_splatting/nodes/gaussian_splat_node_3d.h:73`).

## Usage
<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Primary API</th>
      <th>Implementation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Assign preprocessed asset.</td>
      <td><code>set_splat_asset(asset)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:511</code></td>
    </tr>
    <tr>
      <td>Load from file path (compatibility path).</td>
      <td><code>set_ply_file_path(path)</code>, <code>set_auto_load(enabled)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:496</code></td>
    </tr>
    <tr>
      <td>Push procedural data.</td>
      <td><code>set_splat_data(...)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:548</code></td>
    </tr>
    <tr>
      <td>Run manual updates.</td>
      <td><code>set_update_mode(UPDATE_MODE_MANUAL)</code>, <code>update_splats()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:881</code></td>
    </tr>
    <tr>
      <td>Inspect live metrics.</td>
      <td><code>get_visible_splat_count()</code>, <code>get_statistics()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:988</code></td>
    </tr>
  </tbody>
</table>

## API
### Enums
<table>
  <thead>
    <tr>
      <th>Enum</th>
      <th>Values</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>QualityPreset</code></td>
      <td><code>QUALITY_PERFORMANCE</code>, <code>QUALITY_BALANCED</code>, <code>QUALITY_QUALITY</code>, <code>QUALITY_CUSTOM</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.h:81</code></td>
    </tr>
    <tr>
      <td><code>ViewportUpdateMode</code></td>
      <td><code>UPDATE_MODE_ALWAYS</code>, <code>UPDATE_MODE_WHEN_VISIBLE</code>, <code>UPDATE_MODE_WHEN_PARENT_VISIBLE</code>, <code>UPDATE_MODE_MANUAL</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.h:92</code></td>
    </tr>
    <tr>
      <td><code>DebugDrawMode</code></td>
      <td><code>DEBUG_DRAW_OFF</code>, <code>DEBUG_DRAW_WIREFRAME</code>, <code>DEBUG_DRAW_POINTS</code>, <code>DEBUG_DRAW_HEATMAP</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.h:103</code></td>
    </tr>
  </tbody>
</table>

### Properties
<table>
  <thead>
    <tr>
      <th>Inspector path</th>
      <th>Type</th>
      <th>Accessors</th>
      <th>Notes</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>ply_file_path</code></td>
      <td><code>String</code></td>
      <td><code>set_ply_file_path</code>, <code>get_ply_file_path</code></td>
      <td>Deprecated compatibility path. Accepts <code>.ply</code> and <code>.spz</code> by file hint, but new scene workflows should prefer <code>splat_asset</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:89</code></td>
    </tr>
    <tr>
      <td><code>splat_asset</code></td>
      <td><code>GaussianSplatAsset</code></td>
      <td><code>set_splat_asset</code>, <code>get_splat_asset</code></td>
      <td>Assign a preprocessed resource instead of a file path.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:93</code></td>
    </tr>
    <tr>
      <td><code>auto_load</code></td>
      <td><code>bool</code></td>
      <td><code>set_auto_load</code>, <code>is_auto_load_enabled</code></td>
      <td>Loads automatically only when the node is inside the tree.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:536</code></td>
    </tr>
    <tr>
      <td><code>quality/preset</code></td>
      <td><code>int (QualityPreset)</code></td>
      <td><code>set_quality_preset</code>, <code>get_quality_preset</code></td>
      <td>Preset values are applied through quality helper config.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:913</code></td>
    </tr>
    <tr>
      <td><code>quality/lod_bias</code></td>
      <td><code>float</code></td>
      <td><code>set_lod_bias</code>, <code>get_lod_bias</code></td>
      <td>Clamped to <code>0.1..4.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:798</code></td>
    </tr>
    <tr>
      <td><code>quality/max_render_distance</code></td>
      <td><code>float</code></td>
      <td><code>set_max_render_distance</code>, <code>get_max_render_distance</code></td>
      <td>Clamped to <code>&gt;= 0.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:804</code></td>
    </tr>
    <tr>
      <td><code>quality/max_splat_count</code></td>
      <td><code>int</code></td>
      <td><code>set_max_splat_count</code>, <code>get_max_splat_count</code></td>
      <td>Clamped to <code>&gt;= 1000</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:810</code></td>
    </tr>
    <tr>
      <td><code>painterly/enabled</code></td>
      <td><code>bool</code></td>
      <td><code>set_enable_painterly</code>, <code>is_painterly_enabled</code></td>
      <td>Enables painterly tuning and streaming flags.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:819</code></td>
    </tr>
    <tr>
      <td><code>painterly/edge_threshold</code></td>
      <td><code>float</code></td>
      <td><code>set_edge_threshold</code>, <code>get_edge_threshold</code></td>
      <td>Clamped to <code>0.0..1.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:833</code></td>
    </tr>
    <tr>
      <td><code>painterly/stroke_opacity</code></td>
      <td><code>float</code></td>
      <td><code>set_stroke_opacity</code>, <code>get_stroke_opacity</code></td>
      <td>Clamped to <code>0.0..1.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:842</code></td>
    </tr>
    <tr>
      <td><code>painterly/stroke_width</code></td>
      <td><code>float</code></td>
      <td><code>set_stroke_width</code>, <code>get_stroke_width</code></td>
      <td>Clamped to <code>0.1..5.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:850</code></td>
    </tr>
    <tr>
      <td><code>painterly/temporal_blend</code></td>
      <td><code>float</code></td>
      <td><code>set_temporal_blend</code>, <code>get_temporal_blend</code></td>
      <td>Clamped to <code>0.01..1.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:866</code></td>
    </tr>
    <tr>
      <td><code>painterly/seed</code></td>
      <td><code>int</code></td>
      <td><code>set_painterly_seed</code>, <code>get_painterly_seed</code></td>
      <td>Clamped to <code>0..65535</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:875</code></td>
    </tr>
    <tr>
      <td><code>rendering/update_mode</code></td>
      <td><code>int (ViewportUpdateMode)</code></td>
      <td><code>set_update_mode</code>, <code>get_update_mode</code></td>
      <td>Manual mode disables automatic processing.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:895</code></td>
    </tr>
    <tr>
      <td><code>rendering/cast_shadow</code></td>
      <td><code>bool</code></td>
      <td><code>set_cast_shadow</code>, <code>get_cast_shadow</code></td>
      <td>Applies cast shadow state to render instance.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:902</code></td>
    </tr>
    <tr>
      <td><code>rendering/frustum_culling</code></td>
      <td><code>bool</code></td>
      <td><code>set_use_frustum_culling</code>, <code>is_frustum_culling_enabled</code></td>
      <td>Applies immediately to renderer settings.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:916</code></td>
    </tr>
    <tr>
      <td><code>rendering/opacity</code></td>
      <td><code>float</code></td>
      <td><code>set_opacity</code>, <code>get_opacity</code></td>
      <td>Clamped to <code>0.0..1.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:927</code></td>
    </tr>
    <tr>
      <td><code>rendering/color_grading</code></td>
      <td><code>ColorGradingResource</code></td>
      <td><code>set_color_grading</code>, <code>get_color_grading</code></td>
      <td>Used by real-time grading and baking API.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1788</code></td>
    </tr>
    <tr>
      <td><code>debug/preview_enabled</code></td>
      <td><code>bool</code></td>
      <td><code>set_preview_enabled</code>, <code>is_preview_enabled</code></td>
      <td>Controls editor preview visibility path.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:935</code></td>
    </tr>
    <tr>
      <td><code>debug/show_bounds</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_bounds</code>, <code>is_showing_bounds</code></td>
      <td>Toggles gizmo bounds rendering.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:941</code></td>
    </tr>
    <tr>
      <td><code>debug/show_statistics</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_statistics</code>, <code>is_showing_statistics</code></td>
      <td>Exposes read-only <code>stats/*</code> inspector fields when active.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:486</code></td>
    </tr>
    <tr>
      <td><code>debug/show_tile_grid</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_tile_grid</code>, <code>is_showing_tile_grid</code></td>
      <td>Persists via settings manager and updates renderer when allowed.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:585</code></td>
    </tr>
    <tr>
      <td><code>debug/show_density_heatmap</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_density_heatmap</code>, <code>is_showing_density_heatmap</code></td>
      <td>Persists via settings manager and updates renderer when allowed.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:599</code></td>
    </tr>
    <tr>
      <td><code>debug/show_performance_hud</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_performance_hud</code>, <code>is_showing_performance_hud</code></td>
      <td>Persists via settings manager and updates renderer when allowed.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:613</code></td>
    </tr>
    <tr>
      <td><code>debug/show_lod_spheres</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_lod_spheres</code>, <code>is_showing_lod_spheres</code></td>
      <td>Updates gizmos only.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:631</code></td>
    </tr>
    <tr>
      <td><code>debug/show_performance_overlay</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_performance_overlay</code>, <code>is_showing_performance_overlay</code></td>
      <td>Updates gizmos only.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:640</code></td>
    </tr>
    <tr>
      <td><code>debug/overlay_opacity</code></td>
      <td><code>float</code></td>
      <td><code>set_debug_overlay_opacity</code>, <code>get_debug_overlay_opacity</code></td>
      <td>Clamped to <code>0.0..1.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:649</code></td>
    </tr>
    <tr>
      <td><code>debug/debug_draw_mode</code></td>
      <td><code>int (DebugDrawMode)</code></td>
      <td><code>set_debug_draw_mode</code>, <code>get_debug_draw_mode</code></td>
      <td>Swaps renderer preview mode unless runtime preview override is enabled.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:662</code></td>
    </tr>
    <tr>
      <td><code>debug/runtime_preview</code></td>
      <td><code>bool</code></td>
      <td><code>set_runtime_preview_enabled</code>, <code>is_runtime_preview_enabled</code></td>
      <td>Temporarily forces runtime modification preview mode in renderer.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:682</code></td>
    </tr>
    <tr>
      <td><code>debug/show_residency_hud</code></td>
      <td><code>bool</code></td>
      <td><code>set_show_residency_hud</code>, <code>is_showing_residency_hud</code></td>
      <td>Persists via settings manager and updates renderer when allowed.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:701</code></td>
    </tr>
  </tbody>
</table>

### Methods
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Behavior</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>reload_asset()</code></td>
      <td>Triggers the same load path as setting a new file path.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:544</code></td>
    </tr>
    <tr>
      <td><code>is_asset_loading()</code></td>
      <td>Returns asynchronous load state managed by the asset helper.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_helpers.cpp:90</code></td>
    </tr>
    <tr>
      <td><code>set_splat_data(...)</code></td>
      <td>Builds a runtime asset from arrays after validating all optional array lengths.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:589</code></td>
    </tr>
    <tr>
      <td><code>bake_color_grading()</code></td>
      <td>Bakes grading into data, disables grading resource, and returns <code>Error</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1794</code></td>
    </tr>
    <tr>
      <td><code>restore_color_grading()</code></td>
      <td>Restores original colors and re-enables grading resource if present.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1821</code></td>
    </tr>
    <tr>
      <td><code>is_color_grading_baked()</code></td>
      <td>Reports whether bake state is present in current renderer data.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1837</code></td>
    </tr>
    <tr>
      <td><code>get_visible_splat_count()</code></td>
      <td>Returns node-level visible splat count after sync with renderer stats.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1106</code></td>
    </tr>
    <tr>
      <td><code>get_total_splat_count()</code></td>
      <td>Returns node-level total splat count from asset or procedural data.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1111</code></td>
    </tr>
    <tr>
      <td><code>get_last_update_time_ms()</code></td>
      <td>Returns elapsed update time measured in <code>update_splats()</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1230</code></td>
    </tr>
    <tr>
      <td><code>get_gpu_memory_mb()</code></td>
      <td>Returns estimated GPU memory derived from loaded or procedural buffers.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1129</code></td>
    </tr>
    <tr>
      <td><code>get_statistics()</code></td>
      <td>Returns node counters and merges any renderer stats dictionary keys.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:988</code></td>
    </tr>
    <tr>
      <td><code>get_configuration_warnings()</code></td>
      <td>Returns warnings for missing assets, missing files, zero distance, and non-uniform scale.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1273</code></td>
    </tr>
    <tr>
      <td><code>get_renderer()</code></td>
      <td>Returns the shared renderer instance for the node world.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1779</code></td>
    </tr>
    <tr>
      <td><code>update_splats()</code></td>
      <td>Performs the full render update path and refreshes timing and metrics.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1026</code></td>
    </tr>
    <tr>
      <td><code>force_update()</code></td>
      <td>Calls <code>update_splats()</code> and emits <code>viewport_visibility_changed</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1268</code></td>
    </tr>
  </tbody>
</table>

### Signals
<table>
  <thead>
    <tr>
      <th>Signal</th>
      <th>Parameters</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>asset_loaded</code></td>
      <td>None</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:281</code></td>
    </tr>
    <tr>
      <td><code>asset_loading_failed</code></td>
      <td><code>error: String</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:282</code></td>
    </tr>
    <tr>
      <td><code>viewport_visibility_changed</code></td>
      <td><code>visible: bool</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:283</code></td>
    </tr>
  </tbody>
</table>

## Examples
```gdscript
extends Node3D

@onready var splat: GaussianSplatNode3D = $GaussianSplatNode3D

func _ready() -> void:
    var asset := load("res://splats/scene.ply") as GaussianSplatAsset
    splat.set_splat_asset(asset)
    splat.set_quality_preset(GaussianSplatNode3D.QUALITY_BALANCED)
    splat.set_update_mode(GaussianSplatNode3D.UPDATE_MODE_WHEN_VISIBLE)
```

```gdscript
extends Node3D

@onready var splat: GaussianSplatNode3D = $GaussianSplatNode3D

func _ready() -> void:
    var positions := PackedVector3Array([Vector3.ZERO])
    var colors := PackedColorArray([Color(1.0, 1.0, 1.0, 1.0)])
    splat.set_splat_data(positions, colors)
    splat.set_update_mode(GaussianSplatNode3D.UPDATE_MODE_MANUAL)
    splat.update_splats()
```

## Troubleshooting
<table>
  <thead>
    <tr>
      <th>Problem</th>
      <th>Action</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>No splats appear after scene start.</td>
      <td>Check <code>get_configuration_warnings()</code> for missing asset/path and zero render distance cases.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:1273</code></td>
    </tr>
    <tr>
      <td><code>set_splat_data()</code> does nothing.</td>
      <td>Ensure every optional array has the same length as <code>positions</code> when provided.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:589</code></td>
    </tr>
  </tbody>
</table>
