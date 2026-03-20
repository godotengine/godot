# GaussianSplatWorld3D API Reference

## Purpose
Use `GaussianSplatWorld3D` to render merged multi-asset Gaussian splat worlds from a `GaussianSplatWorld` resource in a `Node3D` scene (`modules/gaussian_splatting/nodes/gaussian_splat_world_3d.h:10`).

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
      <td>Assign a merged world resource.</td>
      <td><code>set_world(world)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:234</code></td>
    </tr>
    <tr>
      <td>Apply world data to the shared renderer.</td>
      <td><code>apply_world()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:300</code></td>
    </tr>
    <tr>
      <td>Remove world data and reset bounds.</td>
      <td><code>clear_world()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:305</code></td>
    </tr>
    <tr>
      <td>Configure quality and LOD parameters.</td>
      <td><code>set_lod_bias(bias)</code>, <code>set_max_render_distance(distance)</code>, <code>set_max_splat_count(count)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:258</code></td>
    </tr>
    <tr>
      <td>Access the shared renderer instance.</td>
      <td><code>get_renderer()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:123</code></td>
    </tr>
  </tbody>
</table>

## API
### Enums

This class does not define any enums.

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
      <td><code>world</code></td>
      <td><code>GaussianSplatWorld</code></td>
      <td><code>set_world</code>, <code>get_world</code></td>
      <td>Resource containing merged Gaussian data, bounds, and static chunks.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:125</code></td>
    </tr>
    <tr>
      <td><code>auto_apply_on_ready</code></td>
      <td><code>bool</code></td>
      <td><code>set_auto_apply_on_ready</code>, <code>is_auto_apply_on_ready</code></td>
      <td>When enabled, calls <code>apply_world()</code> automatically on <code>NOTIFICATION_READY</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:127</code></td>
    </tr>
    <tr>
      <td><code>cast_shadow</code></td>
      <td><code>bool</code></td>
      <td><code>set_cast_shadow</code>, <code>is_cast_shadow</code></td>
      <td>Applies shadow casting setting to the render instance.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:128</code></td>
    </tr>
    <tr>
      <td><code>quality/lod_enabled</code></td>
      <td><code>bool</code></td>
      <td><code>set_lod_enabled</code>, <code>is_lod_enabled</code></td>
      <td>Toggles level-of-detail processing on the renderer.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:131</code></td>
    </tr>
    <tr>
      <td><code>quality/lod_bias</code></td>
      <td><code>float</code></td>
      <td><code>set_lod_bias</code>, <code>get_lod_bias</code></td>
      <td>Clamped to <code>0.1..4.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:132</code></td>
    </tr>
    <tr>
      <td><code>quality/max_render_distance</code></td>
      <td><code>float</code></td>
      <td><code>set_max_render_distance</code>, <code>get_max_render_distance</code></td>
      <td>Clamped to <code>&gt;= 0.0</code>. Suffix <code>m</code> in inspector.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:134</code></td>
    </tr>
    <tr>
      <td><code>quality/max_splat_count</code></td>
      <td><code>int</code></td>
      <td><code>set_max_splat_count</code>, <code>get_max_splat_count</code></td>
      <td>Clamped to <code>&gt;= 1000</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:136</code></td>
    </tr>
    <tr>
      <td><code>rendering/frustum_culling</code></td>
      <td><code>bool</code></td>
      <td><code>set_use_frustum_culling</code>, <code>is_frustum_culling_enabled</code></td>
      <td>Applies immediately to renderer settings when the renderer is valid.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:140</code></td>
    </tr>
    <tr>
      <td><code>rendering/async_upload_enabled</code></td>
      <td><code>bool</code></td>
      <td><code>set_async_upload_enabled</code>, <code>is_async_upload_enabled</code></td>
      <td>Enables asynchronous GPU upload for world data.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:142</code></td>
    </tr>
    <tr>
      <td><code>rendering/opacity</code></td>
      <td><code>float</code></td>
      <td><code>set_opacity</code>, <code>get_opacity</code></td>
      <td>Clamped to <code>0.0..1.0</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:144</code></td>
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
      <td><code>apply_world()</code></td>
      <td>Ensures a renderer exists, then applies the current <code>GaussianSplatWorld</code> resource to the shared renderer, registering ownership and pushing quality/streaming settings.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:300</code></td>
    </tr>
    <tr>
      <td><code>clear_world()</code></td>
      <td>Unregisters the shared renderer, resets bounds to empty, and updates the render instance.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:305</code></td>
    </tr>
    <tr>
      <td><code>get_renderer()</code></td>
      <td>Returns the shared <code>GaussianSplatRenderer</code> instance for the node's World3D.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:123</code></td>
    </tr>
  </tbody>
</table>

### Signals

This class does not define any signals.

## Examples
```gdscript
extends Node3D

@onready var world_node: GaussianSplatWorld3D = $GaussianSplatWorld3D

func _ready() -> void:
    var world_res := load("res://worlds/cityscape.gsplatworld") as GaussianSplatWorld
    world_node.set_world(world_res)
    world_node.set_lod_bias(1.5)
    world_node.set_max_render_distance(500.0)
    world_node.set_opacity(0.9)
    world_node.apply_world()
```

```gdscript
extends Node3D

## Clears the current world and applies a new one at runtime.
@onready var world_node: GaussianSplatWorld3D = $GaussianSplatWorld3D

func swap_world(new_resource: GaussianSplatWorld) -> void:
    world_node.clear_world()
    world_node.set_world(new_resource)
    world_node.set_async_upload_enabled(true)
    world_node.set_use_frustum_culling(true)
    world_node.set_max_splat_count(2000000)
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
      <td>World data does not appear after assigning a resource.</td>
      <td>Ensure <code>auto_apply_on_ready</code> is enabled, or call <code>apply_world()</code> manually after the node enters the tree.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:162</code></td>
    </tr>
    <tr>
      <td>Non-identity transform warning logged at startup.</td>
      <td>Place the <code>GaussianSplatWorld3D</code> node at the scene origin. Merged world data is assumed to be in world space.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:425</code></td>
    </tr>
    <tr>
      <td>Renderer ownership conflict prevents data from loading.</td>
      <td>Only one <code>GaussianSplatWorld3D</code> per World3D should drive the shared renderer. Remove or disable duplicate world nodes.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:455</code></td>
    </tr>
    <tr>
      <td>Zero-splat warning printed and renderer stays disconnected.</td>
      <td>Verify the <code>GaussianSplatWorld</code> resource was exported with splat data. Check the source <code>GaussianSplatContainer</code> merge result.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world_3d.cpp:498</code></td>
    </tr>
  </tbody>
</table>
