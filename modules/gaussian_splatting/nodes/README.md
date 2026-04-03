# GaussianSplatNode3D

_Last updated: 2025-11-27._

A 3D node for rendering Gaussian splats in Godot scenes with full editor integration.

> **New in this repository:** `GaussianSplatContainer` groups multiple `GaussianSplatNode3D` instances and bakes their data into a shared world-space `GaussianData`, splitting the merged buffer into culling-friendly chunks for static scenes.

### GaussianSplatContainer quick usage

```
# Merge children into world-space data, then apply to a renderer node.
var container = $GaussianSplatContainer
container.merge_children()
container.apply_to_node($MergedWorldRenderer)  # GaussianSplatNode3D
```

Methods:
- `merge_children()` – build merged data + chunk cache.
- `apply_to_renderer(renderer)` – legacy low-level convenience helper that pushes merged data + static chunks into a renderer. Prefer `apply_to_node(node)` or `export_world_resource()` for normal scene workflows.
- `apply_to_node(node)` – convenience wrapper for GaussianSplatNode3D and GaussianSplatWorld3D targets.
- `merge_children_to_node(node)` – merge + apply in one step.
- `export_world_resource()` – build a GaussianSplatWorld resource from merged data.

Note: `apply_to_renderer()` bypasses the canonical node/world submission flow and
is kept only as a low-level convenience while direct raw-data renderer binding
still exists.

Optional properties:
- `apply_to_target_on_merge`: auto-apply after merge.
- `target_node_path`: target GaussianSplatNode3D to receive merged data.

CLI bake (headless):

```
godot --headless --path <project> --script scripts/bake_gsplatworld.gd -- \
  --scene=res://scenes/world.tscn --container=/root/World/GaussianSplatContainer \
  --output=res://worlds/world.gsplatworld --chunk_size=25.0
```

Workflow guide: `docs/workflows/GSPLATWORLD_BAKE.md`

## GaussianSplatWorld3D (unified world renderer)

Node intended for **single-renderer** large scenes. It consumes a
`GaussianSplatWorld` resource (merged data + static chunks) and renders it
through one GaussianSplatRenderer instance.

Quick usage:
```
var container = $GaussianSplatContainer
container.merge_children()

var world = container.export_world_resource()
$GaussianSplatWorld3D.world = world
$GaussianSplatWorld3D.apply_world()
```

Notes:
- The world asset is assumed to be in world space; keep the node transform at
  identity to avoid double transforms.
- This node is the recommended path for large static scenes (global sort per
  renderer).

## GaussianSplatDynamicInstance3D (instance registry)

Lightweight node that registers a `GaussianSplatAsset` with the instance registry so the shared renderer can stream and draw it alongside other instances. It does not render by itself; keep at least one `GaussianSplatNode3D` or `GaussianSplatWorld3D` in the scene to own the renderer.

Quick usage:
```
var dynamic = GaussianSplatDynamicInstance3D.new()
dynamic.splat_asset = preload("res://assets/dynamic_splat_asset.tres")
add_child(dynamic)
```

Properties:
- `splat_asset`: GaussianSplatAsset used by the instance registry (required for rendering).
- `gaussian_data`: Optional GaussianData resource for authoring; must be converted to an asset for runtime rendering.
- `ply_file_path`: Deprecated compatibility path that loads directly into GaussianData. Prefer `splat_asset` or explicit `gaussian_data`.
- `auto_load`: Load asset/path on enter tree when true.

Methods:
- `reload_asset()` – (re)load and register the instance.
- `register_instance()` / `unregister_instance()` – manual control.

Notes:
- Visibility toggles registration; hidden nodes are removed from the instance registry.
- Requires the instance pipeline to be enabled; otherwise the node will warn and no-op.

## Features

- **Scene Integration**: Assign GaussianSplatAsset resources or drag and drop Gaussian splat files for asset-backed scene setup
- **Format Support**: Import PLY and SPZ (Niantic compressed) formats
- **Quality Presets**: Performance, Balanced, Quality, and Custom quality settings
- **Painterly Rendering**: Advanced artistic rendering with configurable strokes
- **LOD Support**: Automatic level-of-detail based on distance
- **Editor Gizmos**: Visual bounds, LOD radius, and statistics display
- **Performance Monitoring**: Real-time metrics in the inspector

## Usage

### Adding to Scene

1. In the scene dock, click "Add Node"
2. Search for "GaussianSplatNode3D"
3. Add the node to your scene
4. Assign a `GaussianSplatAsset`, or drag a Gaussian splat file (`.ply` or `.spz`) onto the node to create an asset-backed assignment

### Properties

#### File Management
- `splat_asset`: Reference to a GaussianSplatAsset resource (preferred)
- `ply_file_path`: Deprecated compatibility path to the Gaussian splat file (.ply or .spz)
- `auto_load`: Automatically load the file when the compatibility path is set

> **Note:** The property is named `ply_file_path` for historical reasons but accepts both PLY and SPZ formats.

#### Quality Settings
- `quality/preset`: Choose from predefined quality levels
  - **Performance**: Fast rendering, reduced budgets (200K splats max)
  - **Balanced**: General-purpose quality/performance trade-off (500K splats max)
  - **Quality**: Highest fidelity with aggressive budgets (1M splats max)
  - **Custom**: Manual control over all parameters
- `quality/lod_bias`: Adjust level-of-detail aggressiveness (0.1-4.0)
- `quality/max_render_distance`: Maximum distance for rendering
- `quality/max_splat_count`: Maximum number of splats to render
  - These quality settings also feed streaming overrides (prefetch distance,
    VRAM budget, and LOD distances) when streaming is enabled.

#### Painterly Settings
- `painterly/enabled`: Enable artistic rendering style
- `painterly/edge_threshold`: Edge detection sensitivity (0.0-1.0)
- `painterly/stroke_opacity`: Opacity of paint strokes (0.0-1.0)
- `painterly/stroke_width`: Width of paint strokes (0.1-5.0)
- `painterly/temporal_blend`: Temporal smoothing factor (0.0-1.0)
- `painterly/seed`: Random seed for painterly effects

#### Rendering Settings
- `rendering/update_mode`: When to update the splats
  - **Always**: Update every frame
  - **When Visible**: Update only when visible
  - **When Parent Visible**: Update when parent is visible
  - **Manual**: Update only when requested
- `rendering/cast_shadow`: Enable shadow casting
- `rendering/frustum_culling`: Enable frustum culling optimization
- `rendering/opacity`: Overall opacity of the splats (0.0-1.0)

#### Debug Settings
- `debug/preview_enabled`: Show preview in editor viewport
  - When **enabled** (default): Splats render in editor viewports
  - When **disabled**: Splats are hidden in editor (gizmos still show bounds)
  - **Note:** At runtime, this setting is ignored - splats always render when visible
- `debug/show_bounds`: Display bounding box gizmo
- `debug/show_statistics`: Show performance statistics
- `debug/dump_gpu_counters`: Log GPU counter snapshots asynchronously (off by default)

##### Visual Debug Overlays
These overlays render directly in the viewport when using the tile renderer:
- `debug/show_tile_grid`: Overlay tile boundaries on the rendered image
- `debug/show_density_heatmap`: Overlay splat density heatmap (per-tile)
- `debug/overlay_opacity`: Opacity of tile grid and heatmap overlays (0.0-1.0, default 0.3)

##### Stats-Only Toggles (No Viewport Rendering)
These flags populate `get_statistics()` but do **not** draw HUD text in the viewport:
- `debug/show_performance_hud`: Enables performance metrics in stats dictionary
- `debug/show_residency_hud`: Enables memory residency info in stats dictionary

> **Note:** The HUD toggles affect what data is collected and returned by
> `get_statistics()`, but there is currently no in-viewport text rendering.
> Use the inspector's statistics panel or call `get_statistics()` to view this data.

##### Other Debug Options
- `debug/show_lod_spheres`: Display LOD distance spheres as gizmos
- `debug/show_performance_overlay`: Show performance overlay gizmo
- `debug/debug_draw_mode`: Visualization mode (Off, Wireframe, Points, Heatmap)
- `debug/runtime_preview`: Enable debug visualization at runtime

### Methods

```gdscript
# Reload the currently configured asset or compatibility file path
reload_asset()

# Force update the splats
force_update()

# Manual update (when update_mode is Manual)
update_splats()

# Get performance statistics
var stats = get_statistics()
print("Visible splats: ", stats.visible_splats)
print("Total splats: ", stats.total_splats)
print("Update time: ", stats.update_time_ms)
print("GPU memory: ", stats.gpu_memory_mb)
```

### Signals

- `asset_loaded`: Emitted when PLY file is successfully loaded
- `asset_loading_failed(error)`: Emitted when loading fails
- `viewport_visibility_changed(visible)`: Emitted when visibility state changes

## Editor Integration

### Gizmos

The node provides several visual gizmos in the editor:

1. **Bounding Box**: Shows the spatial bounds of the splat data
2. **LOD Radius**: Displays concentric circles for LOD distances
3. **Statistics Overlay**: Shows splat count and performance metrics
4. **Preview Points**: Displays a subset of splats for visualization

### Inspector

The inspector provides:
- Organized property groups
- Real-time statistics display
- Quality preset dropdown
- Drag-and-drop support for PLY and SPZ files
- Context-sensitive property visibility

### Drag and Drop

You can drag Gaussian splat files (.ply, .spz) from:
- FileSystem dock directly onto the node
- External file explorer onto the node in the scene
- Inspector file path field

## Performance Tips

1. **Use Quality Presets**: Start with Balanced and adjust as needed
2. **Enable Culling**: Both frustum and occlusion culling improve performance
3. **Adjust LOD Bias**: Higher values reduce quality but improve performance
4. **Limit Max Splats**: Set reasonable limits based on target hardware
5. **Update Mode**: Use "When Visible" for static splats
6. **Distance Culling**: Set max_render_distance for far objects

## Example Code

```gdscript
extends Node3D

func _ready():
    # Create a Gaussian splat node
    var splat = GaussianSplatNode3D.new()
    splat.name = "MyGaussianSplat"
    add_child(splat)

    # Configure quality
    splat.set_quality_preset(GaussianSplatNode3D.QUALITY_QUALITY)
    splat.set_max_render_distance(100.0)

    # Enable painterly rendering
    splat.set_enable_painterly(true)
    splat.set_stroke_opacity(0.8)
    splat.set_edge_threshold(0.15)

    # Load an asset resource (preferred)
    splat.set_splat_asset(load("res://assets/my_splat.ply"))

    # Connect to signals
    splat.asset_loaded.connect(_on_splat_loaded)
    splat.asset_loading_failed.connect(_on_splat_load_failed)

func _on_splat_loaded():
    print("Splat loaded successfully!")

func _on_splat_load_failed(error):
    print("Failed to load splat: ", error)
```

## Technical Details

### Memory Management
- Splat data is stored in GPU buffers for efficient rendering
- Automatic cleanup when node is freed
- Shared resources between multiple instances

### Rendering Pipeline
1. Gaussian splat file loaded and parsed (PLY or SPZ)
2. Splats uploaded to GPU buffers
3. View-dependent sorting performed
4. LOD selection based on distance
5. Painterly effects applied (if enabled)
6. Final rendering to viewport

### Supported Formats

**PLY (Polygon File Format)**
- Standard format for Gaussian Splatting data
- Supports ASCII and binary encoding
- Full property validation with warnings for missing optional data
- Recommended for maximum compatibility and debugging

**SPZ (Niantic Compressed Format)**
- Compressed format providing ~10x size reduction over PLY
- Used by Scaniverse and mobile applications
- Fixed-point position encoding with configurable precision
- Smallest-three quaternion rotation encoding
- Gzip-compressed data stream

### Limitations
- Maximum splat count depends on GPU memory
- Sorting performance scales with splat count
- Real-time editing not yet supported
- Shadow casting implementation pending
