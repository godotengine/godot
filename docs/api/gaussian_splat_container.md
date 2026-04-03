# GaussianSplatContainer API Reference

## Purpose
Use `GaussianSplatContainer` to merge multiple child `GaussianSplatNode3D` nodes into a single chunk-based Gaussian splat dataset using spatial subdivision (`modules/gaussian_splatting/nodes/gaussian_splat_container.h:15`).

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
      <td>Merge child splat nodes on scene start.</td>
      <td><code>set_merge_on_ready(true)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:44</code></td>
    </tr>
    <tr>
      <td>Trigger a merge manually.</td>
      <td><code>merge_children()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:71</code></td>
    </tr>
    <tr>
      <td>Export merged data as a reusable world resource.</td>
      <td><code>export_world_resource()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:145</code></td>
    </tr>
    <tr>
      <td>Apply merged data to a target node in one call.</td>
      <td><code>merge_children_to_node(node)</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:140</code></td>
    </tr>
    <tr>
      <td>Inspect merge results.</td>
      <td><code>get_chunk_count()</code>, <code>get_chunk_sizes()</code>, <code>get_chunk_aabbs()</code></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:226</code></td>
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
      <td><code>merge_on_ready</code></td>
      <td><code>bool</code></td>
      <td><code>set_merge_on_ready</code>, <code>is_merge_on_ready</code></td>
      <td>When enabled, automatically calls <code>merge_children()</code> on <code>NOTIFICATION_READY</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:34</code></td>
    </tr>
    <tr>
      <td><code>chunk_size</code></td>
      <td><code>float</code></td>
      <td><code>set_chunk_size</code>, <code>get_chunk_size</code></td>
      <td>Spatial subdivision cell size for merge chunking. Clamped to <code>&gt;= 0.1</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:35</code></td>
    </tr>
    <tr>
      <td><code>hide_children_after_merge</code></td>
      <td><code>bool</code></td>
      <td><code>set_hide_children_after_merge</code>, <code>get_hide_children_after_merge</code></td>
      <td>Hides child <code>GaussianSplatNode3D</code> nodes after a merge completes. Restores visibility on <code>clear_merged_data()</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:36</code></td>
    </tr>
    <tr>
      <td><code>apply_to_target_on_merge</code></td>
      <td><code>bool</code></td>
      <td><code>set_apply_to_target_on_merge</code>, <code>is_apply_to_target_on_merge</code></td>
      <td>When enabled, automatically calls <code>apply_to_node()</code> on the <code>target_node_path</code> node after each merge.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:37</code></td>
    </tr>
    <tr>
      <td><code>target_node_path</code></td>
      <td><code>NodePath</code></td>
      <td><code>set_target_node_path</code>, <code>get_target_node_path</code></td>
      <td>Path to a <code>GaussianSplatWorld3D</code> or <code>GaussianSplatNode3D</code> that receives merged data when <code>apply_to_target_on_merge</code> is enabled.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:38</code></td>
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
      <td><code>merge_children()</code></td>
      <td>Collects all child <code>GaussianSplatNode3D</code> nodes with valid assets, merges their splat data using chunk-based spatial subdivision, optionally hides children and applies to the target node.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:71</code></td>
    </tr>
    <tr>
      <td><code>clear_merged_data()</code></td>
      <td>Clears chunks, resizes merged data to zero, resets bounds, and restores child visibility if <code>hide_children_after_merge</code> is enabled.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:159</code></td>
    </tr>
    <tr>
      <td><code>apply_to_renderer(renderer)</code></td>
      <td>Low-level compatibility helper that pushes merged <code>GaussianData</code> and static chunks to the given <code>GaussianSplatRenderer</code>. Prefer <code>apply_to_node()</code> or <code>export_world_resource()</code> for normal scene workflows. Returns <code>ERR_UNAVAILABLE</code> when no merged data exists.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:86</code></td>
    </tr>
    <tr>
      <td><code>apply_to_node(node)</code></td>
      <td>Applies merged data to a <code>GaussianSplatWorld3D</code> (via <code>export_world_resource()</code>) or a <code>GaussianSplatNode3D</code> (via asset conversion). Returns <code>ERR_INVALID_PARAMETER</code> for unsupported node types.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:102</code></td>
    </tr>
    <tr>
      <td><code>merge_children_to_node(node)</code></td>
      <td>Convenience method that calls <code>merge_children()</code> followed by <code>apply_to_node(node)</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:140</code></td>
    </tr>
    <tr>
      <td><code>export_world_resource()</code></td>
      <td>Creates and returns a new <code>GaussianSplatWorld</code> resource populated with the merged data, bounds, and static chunks. Returns null when no merged data is available.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:145</code></td>
    </tr>
    <tr>
      <td><code>get_merged_data()</code></td>
      <td>Returns the current merged <code>GaussianData</code> resource, or null if no merge has been performed.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:29</code></td>
    </tr>
    <tr>
      <td><code>get_chunk_count()</code></td>
      <td>Returns the number of spatial chunks produced by the most recent merge.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:30</code></td>
    </tr>
    <tr>
      <td><code>get_chunk_sizes()</code></td>
      <td>Returns a <code>PackedInt32Array</code> where each element is the index count of the corresponding chunk.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:226</code></td>
    </tr>
    <tr>
      <td><code>get_chunk_aabbs()</code></td>
      <td>Returns an <code>Array</code> of <code>AABB</code> values, one per chunk, representing each chunk's bounding box.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:235</code></td>
    </tr>
  </tbody>
</table>

### Signals

This class does not define any signals.

## Examples
```gdscript
extends Node3D

## Merge children at startup and push to a GaussianSplatWorld3D target.
@onready var container: GaussianSplatContainer = $GaussianSplatContainer
@onready var world_node: GaussianSplatWorld3D = $GaussianSplatWorld3D

func _ready() -> void:
    container.set_chunk_size(16.0)
    container.set_hide_children_after_merge(true)
    container.merge_children()
    container.apply_to_node(world_node)
    print("Chunks: ", container.get_chunk_count())
```

```gdscript
extends Node3D

## Export merged data to a GaussianSplatWorld resource and save to disk.
@onready var container: GaussianSplatContainer = $GaussianSplatContainer

func export_merged_world(save_path: String) -> void:
    container.merge_children()
    var world_resource := container.export_world_resource()
    if world_resource == null:
        push_warning("No merged data to export.")
        return
    ResourceSaver.save(world_resource, save_path)
    print("Saved ", container.get_chunk_count(), " chunks to ", save_path)
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
      <td>No merged data after calling <code>merge_children()</code>.</td>
      <td>Verify that child <code>GaussianSplatNode3D</code> nodes have valid <code>splat_asset</code> resources with non-zero splat counts.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:191</code></td>
    </tr>
    <tr>
      <td><code>apply_to_node()</code> returns <code>ERR_INVALID_PARAMETER</code>.</td>
      <td>The target node must be a <code>GaussianSplatWorld3D</code> or <code>GaussianSplatNode3D</code>. Other node types are not supported.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:120</code></td>
    </tr>
    <tr>
      <td>Target world node warns about non-identity transform.</td>
      <td>Merged splat data is in world space. Keep the target <code>GaussianSplatWorld3D</code> at the scene origin with an identity transform.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:110</code></td>
    </tr>
    <tr>
      <td><code>target_node_path</code> not found warning during auto-apply.</td>
      <td>Ensure the <code>target_node_path</code> points to a valid node in the scene tree when <code>apply_to_target_on_merge</code> is enabled.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp:79</code></td>
    </tr>
  </tbody>
</table>
