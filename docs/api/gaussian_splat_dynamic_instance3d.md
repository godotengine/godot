# GaussianSplatDynamicInstance3D API Reference

## Purpose
Use `GaussianSplatDynamicInstance3D` as a lightweight instance node that registers splat data into the streaming renderer's instance pipeline without creating a dedicated renderer (`modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.h:20`).

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
      <td>Assign a preprocessed asset.</td>
      <td><code>set_splat_asset(asset)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:165</code></td>
    </tr>
    <tr>
      <td>Load splats from a file path (compatibility path).</td>
      <td><code>set_ply_file_path(path)</code>, <code>set_auto_load(true)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:157</code></td>
    </tr>
    <tr>
      <td>Push procedural data directly.</td>
      <td><code>set_gaussian_data(data)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:184</code></td>
    </tr>
    <tr>
      <td>Reload and re-register after changes.</td>
      <td><code>reload_asset()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:203</code></td>
    </tr>
    <tr>
      <td>Control instance lifecycle manually.</td>
      <td><code>register_instance()</code>, <code>unregister_instance()</code>, <code>is_registered()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:230</code></td>
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
      <td><code>ply_file_path</code></td>
      <td><code>String</code></td>
      <td><code>set_ply_file_path</code>, <code>get_ply_file_path</code></td>
      <td>Deprecated compatibility path to a <code>.ply</code> or <code>.spz</code> file. Prefer <code>splat_asset</code> or <code>gaussian_data</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:17</code></td>
    </tr>
    <tr>
      <td><code>splat_asset</code></td>
      <td><code>GaussianSplatAsset</code></td>
      <td><code>set_splat_asset</code>, <code>get_splat_asset</code></td>
      <td>Preprocessed asset resource. Connects to the asset's <code>changed</code> signal for auto-reload.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:21</code></td>
    </tr>
    <tr>
      <td><code>gaussian_data</code></td>
      <td><code>GaussianData</code></td>
      <td><code>set_gaussian_data</code>, <code>get_gaussian_data</code></td>
      <td>Direct data injection. Unrefs automatically if the provided data has zero splats.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:26</code></td>
    </tr>
    <tr>
      <td><code>auto_load</code></td>
      <td><code>bool</code></td>
      <td><code>set_auto_load</code>, <code>is_auto_load_enabled</code></td>
      <td>When enabled, automatically loads and registers data on tree entry and when properties change.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:31</code></td>
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
      <td>Re-evaluates the data source priority: <code>gaussian_data</code> first, then <code>splat_asset</code>, then <code>ply_file_path</code>. Registers or unregisters the instance based on the result.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:203</code></td>
    </tr>
    <tr>
      <td><code>register_instance()</code></td>
      <td>Manually registers this instance with the scene director's instance pipeline. Requires the node to be in the tree, visible, and have valid data.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:230</code></td>
    </tr>
    <tr>
      <td><code>unregister_instance()</code></td>
      <td>Manually unregisters this instance from the scene director.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:234</code></td>
    </tr>
    <tr>
      <td><code>is_registered()</code></td>
      <td>Returns whether this instance is currently registered with the scene director's instance pipeline.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:36</code></td>
    </tr>
  </tbody>
</table>

### Signals

This class does not define any signals.

## Examples
```gdscript
extends Node3D

## Spawn a dynamic splat instance from a preloaded asset.
func spawn_splat(asset: GaussianSplatAsset, spawn_position: Vector3) -> void:
    var instance := GaussianSplatDynamicInstance3D.new()
    instance.set_splat_asset(asset)
    instance.position = spawn_position
    add_child(instance)
```

```gdscript
extends Node3D

## Inject procedural GaussianData and manage registration manually.
@onready var instance: GaussianSplatDynamicInstance3D = $GaussianSplatDynamicInstance3D

func apply_procedural_data(data: GaussianData) -> void:
    instance.set_auto_load(false)
    instance.set_gaussian_data(data)
    instance.register_instance()
    print("Registered: ", instance.is_registered())

func remove_from_pipeline() -> void:
    instance.unregister_instance()
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
      <td>Instance not visible after adding to the scene.</td>
      <td>Confirm the node is inside the tree, inside the world, and visible. The instance pipeline requires all three conditions to register.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:101</code></td>
    </tr>
    <tr>
      <td><code>set_gaussian_data()</code> does not register the instance.</td>
      <td>Ensure the <code>GaussianData</code> resource has a non-zero splat count. Data with zero splats is automatically unref'd.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:187</code></td>
    </tr>
    <tr>
      <td>Warning about missing asset or data for instance pipeline.</td>
      <td>Provide at least one data source. Prefer <code>splat_asset</code> or <code>gaussian_data</code>; the deprecated <code>ply_file_path</code> path still works for compatibility.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:130</code></td>
    </tr>
    <tr>
      <td>File load failure logged with error code.</td>
      <td>If using the deprecated <code>ply_file_path</code> compatibility path, verify it points to an existing, readable file. Check the error code in the log message for details.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance_3d.cpp:284</code></td>
    </tr>
  </tbody>
</table>
