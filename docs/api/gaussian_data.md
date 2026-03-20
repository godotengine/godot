# GaussianData API Reference

## Purpose
Use `GaussianData` to store, manipulate, and query Gaussian splat point-cloud data on the CPU side (`modules/gaussian_splatting/core/gaussian_data.h:251`). It serves as the primary data container for positions, colors, opacities, scales, rotations, spherical harmonics, and painterly metadata, and provides spatial acceleration, animation integration, runtime painting overlays, and GPU buffer management.

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
      <td>Load point cloud from disk.</td>
      <td><code>load_from_file(path)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:112</code></td>
    </tr>
    <tr>
      <td>Populate from packed arrays.</td>
      <td><code>resize(count)</code>, <code>set_positions(...)</code>, <code>set_opacities(...)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:84</code></td>
    </tr>
    <tr>
      <td>Build spatial acceleration.</td>
      <td><code>build_octree()</code>, <code>query_octree(bounds)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:116</code></td>
    </tr>
    <tr>
      <td>Paint splats at runtime.</td>
      <td><code>apply_brush_stroke(...)</code>, <code>commit_runtime_changes()</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:153</code></td>
    </tr>
    <tr>
      <td>Play animated splats.</td>
      <td><code>set_animation_state_machine(anim)</code>, <code>update_animation(delta)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:125</code></td>
    </tr>
    <tr>
      <td>Read statistics.</td>
      <td><code>get_count()</code>, <code>get_aabb()</code>, <code>get_memory_usage()</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:120</code></td>
    </tr>
  </tbody>
</table>

## API

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
      <td><code>2d_mode</code></td>
      <td><code>bool</code></td>
      <td><code>set_2d_mode</code>, <code>get_2d_mode</code></td>
      <td>Enables 2D Gaussian (surfel) rendering where splats use normals as disc orientation.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:141</code></td>
    </tr>
    <tr>
      <td><code>count</code></td>
      <td><code>int</code></td>
      <td>(read-only) <code>get_count</code></td>
      <td>Number of Gaussians in the resource. Range: <code>0..10000000</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:142</code></td>
    </tr>
    <tr>
      <td><code>animation_enabled</code></td>
      <td><code>bool</code></td>
      <td><code>set_animation_enabled</code>, <code>is_animation_enabled</code></td>
      <td>Controls whether animation playback updates are applied.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:143</code></td>
    </tr>
  </tbody>
</table>

### Methods

#### Core Data Management
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
      <td><code>resize(count: int)</code></td>
      <td>Resizes internal Gaussian storage. All entries are default-initialized; existing data is not preserved.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:84</code></td>
    </tr>
    <tr>
      <td><code>get_count() -> int</code></td>
      <td>Returns the number of Gaussians currently stored.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:120</code></td>
    </tr>
    <tr>
      <td><code>get_aabb() -> AABB</code></td>
      <td>Returns the axis-aligned bounding box encompassing all Gaussian positions.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:121</code></td>
    </tr>
    <tr>
      <td><code>get_memory_usage() -> float</code></td>
      <td>Returns estimated memory usage in bytes for the stored Gaussian data.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:122</code></td>
    </tr>
  </tbody>
</table>

#### Batch Operations
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
      <td><code>set_positions(positions: PackedVector3Array)</code></td>
      <td>Sets positions for all Gaussians. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:90</code></td>
    </tr>
    <tr>
      <td><code>set_scales(scales: PackedVector3Array)</code></td>
      <td>Sets per-axis scale vectors for all Gaussians. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:91</code></td>
    </tr>
    <tr>
      <td><code>set_rotations(rotations: TypedArray[Quaternion])</code></td>
      <td>Sets orientation quaternions for all Gaussians. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:92</code></td>
    </tr>
    <tr>
      <td><code>set_opacities(opacities: PackedFloat32Array)</code></td>
      <td>Sets opacity values in [0, 1] for all Gaussians. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:93</code></td>
    </tr>
    <tr>
      <td><code>set_spherical_harmonics(sh_data: PackedFloat32Array)</code></td>
      <td>Sets spherical harmonics coefficients for all Gaussians. Layout depends on the SH degree set by the file loader.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:94</code></td>
    </tr>
    <tr>
      <td><code>get_spherical_harmonics(splat_idx: int) -> PackedFloat32Array</code></td>
      <td>Retrieves all SH coefficients for a single Gaussian at the given index.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:95</code></td>
    </tr>
    <tr>
      <td><code>has_full_sh() -> bool</code></td>
      <td>Returns <code>true</code> if higher-order spherical harmonics coefficients are available beyond DC.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:96</code></td>
    </tr>
    <tr>
      <td><code>get_sh_degree() -> int</code></td>
      <td>Returns the spherical harmonics degree (0-3) of the loaded data.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:97</code></td>
    </tr>
    <tr>
      <td><code>set_palette_ids(palette_ids: PackedInt32Array)</code></td>
      <td>Sets palette lookup indices for painterly color mapping. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:98</code></td>
    </tr>
    <tr>
      <td><code>set_painterly_flags(painterly_flags: PackedInt32Array)</code></td>
      <td>Sets painterly flag bitfields for stylized rendering. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:99</code></td>
    </tr>
    <tr>
      <td><code>get_brush_override_ids() -> PackedInt32Array</code></td>
      <td>Returns 16-bit brush override IDs stored in the painterly flags lane, one per splat.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:100</code></td>
    </tr>
    <tr>
      <td><code>get_brush_override_ids_buffer() -> PackedInt32Array</code></td>
      <td>Returns sanitized (clamped) brush override IDs. Equivalent to <code>get_brush_override_ids()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:101</code></td>
    </tr>
    <tr>
      <td><code>set_brush_override_ids(brush_override_ids: PackedInt32Array)</code></td>
      <td>Sets brush override IDs using the painterly flags storage lane. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:102</code></td>
    </tr>
    <tr>
      <td><code>set_brush_axes(brush_axes: PackedVector2Array)</code></td>
      <td>Sets brush axis orientation vectors for painterly stroke direction. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:103</code></td>
    </tr>
    <tr>
      <td><code>set_stroke_ages(stroke_ages: PackedFloat32Array)</code></td>
      <td>Sets stroke age metadata for painterly rendering animation. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:104</code></td>
    </tr>
    <tr>
      <td><code>set_normals(normals: PackedVector3Array)</code></td>
      <td>Sets per-splat surface normals for 2D Gaussian (surfel) rendering. Array size must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:109</code></td>
    </tr>
  </tbody>
</table>

#### 2D Mode
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
      <td><code>set_2d_mode(enabled: bool)</code></td>
      <td>Enables or disables 2D Gaussian (surfel) mode, where splats render as oriented discs using normals.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:107</code></td>
    </tr>
    <tr>
      <td><code>get_2d_mode() -> bool</code></td>
      <td>Returns <code>true</code> if 2D surfel mode is active.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:108</code></td>
    </tr>
  </tbody>
</table>

#### File I/O
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
      <td><code>load_from_file(path: String) -> Error</code></td>
      <td>Loads Gaussian data from a <code>.ply</code> or <code>.spz</code> file. Returns <code>OK</code> on success or an error code on failure.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:112</code></td>
    </tr>
    <tr>
      <td><code>save_to_file(path: String) -> Error</code></td>
      <td>Saves current Gaussian data to a PLY file. Returns <code>OK</code> on success or an error code on failure.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:113</code></td>
    </tr>
  </tbody>
</table>

#### Spatial Queries
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
      <td><code>build_octree(max_depth: int = 8, min_gaussians: int = 32)</code></td>
      <td>Builds or rebuilds the internal octree for spatial queries. Subdivision stops when a node has fewer than <code>min_gaussians</code> splats or reaches <code>max_depth</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:116</code></td>
    </tr>
    <tr>
      <td><code>query_octree(bounds: AABB) -> TypedArray[int]</code></td>
      <td>Returns an array of Gaussian indices that may intersect the given axis-aligned bounding box.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:117</code></td>
    </tr>
  </tbody>
</table>

#### Animation
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
      <td><code>set_animation_state_machine(animation: GaussianAnimationStateMachine)</code></td>
      <td>Assigns an animation state machine resource for animated splat playback.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:125</code></td>
    </tr>
    <tr>
      <td><code>get_animation_state_machine() -> GaussianAnimationStateMachine</code></td>
      <td>Returns the currently assigned animation state machine, or <code>null</code> if none.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:126</code></td>
    </tr>
    <tr>
      <td><code>has_animation() -> bool</code></td>
      <td>Returns <code>true</code> if an animation state machine is assigned.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:127</code></td>
    </tr>
    <tr>
      <td><code>update_animation(delta: float)</code></td>
      <td>Advances animation state by <code>delta</code> seconds and refreshes the animation cache.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:128</code></td>
    </tr>
    <tr>
      <td><code>apply_animation_at_time(time: float)</code></td>
      <td>Applies the animation state at a specific absolute time in seconds.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:129</code></td>
    </tr>
    <tr>
      <td><code>set_animation_enabled(enabled: bool)</code></td>
      <td>Enables or disables animation playback on this data resource.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:130</code></td>
    </tr>
    <tr>
      <td><code>is_animation_enabled() -> bool</code></td>
      <td>Returns <code>true</code> if animation playback is enabled.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:131</code></td>
    </tr>
    <tr>
      <td><code>set_incremental_saver(saver: GaussianIncrementalSaver)</code></td>
      <td>Assigns an incremental saver resource for progressive save operations.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:132</code></td>
    </tr>
    <tr>
      <td><code>get_incremental_saver() -> GaussianIncrementalSaver</code></td>
      <td>Returns the currently assigned incremental saver, or <code>null</code> if none.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:133</code></td>
    </tr>
    <tr>
      <td><code>get_animated_position(index: int, time: float = -1.0) -> Vector3</code></td>
      <td>Returns the animated position for the Gaussian at <code>index</code>. Pass <code>-1.0</code> for <code>time</code> to use the current cached animation time. Falls back to the base position when no animation is active.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:134</code></td>
    </tr>
    <tr>
      <td><code>get_animated_color(index: int, time: float = -1.0) -> Color</code></td>
      <td>Returns the animated color for the Gaussian at <code>index</code>. Falls back to the base color when no animation is active.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:135</code></td>
    </tr>
    <tr>
      <td><code>get_animated_opacity(index: int, time: float = -1.0) -> float</code></td>
      <td>Returns the animated opacity for the Gaussian at <code>index</code>. Falls back to the base opacity when no animation is active.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:136</code></td>
    </tr>
    <tr>
      <td><code>get_animated_scale(index: int, time: float = -1.0) -> Vector3</code></td>
      <td>Returns the animated scale for the Gaussian at <code>index</code>. Falls back to the base scale when no animation is active.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:137</code></td>
    </tr>
    <tr>
      <td><code>get_animated_rotation(index: int, time: float = -1.0) -> Quaternion</code></td>
      <td>Returns the animated rotation for the Gaussian at <code>index</code>. Falls back to the base rotation when no animation is active.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:138</code></td>
    </tr>
  </tbody>
</table>

#### Runtime Painting
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
      <td><code>set_runtime_position(index: int, position: Vector3)</code></td>
      <td>Sets a non-destructive position override for one Gaussian. The override is applied as an overlay until committed or reverted.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:146</code></td>
    </tr>
    <tr>
      <td><code>set_runtime_color(index: int, color: Color)</code></td>
      <td>Sets a non-destructive color override for one Gaussian.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:147</code></td>
    </tr>
    <tr>
      <td><code>set_runtime_opacity(index: int, opacity: float)</code></td>
      <td>Sets a non-destructive opacity override for one Gaussian. Value should be in [0, 1].</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:148</code></td>
    </tr>
    <tr>
      <td><code>apply_color_range(start: int, count: int, color: Color)</code></td>
      <td>Applies a color to a contiguous range of Gaussians starting at <code>start</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:149</code></td>
    </tr>
    <tr>
      <td><code>mark_range_dirty(start: int, count: int)</code></td>
      <td>Marks a contiguous range of Gaussians as needing GPU re-upload.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:150</code></td>
    </tr>
    <tr>
      <td><code>commit_runtime_changes()</code></td>
      <td>Writes all runtime modification overlays permanently into the base Gaussian data.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:151</code></td>
    </tr>
    <tr>
      <td><code>revert_runtime_changes()</code></td>
      <td>Discards all runtime modification overlays, restoring the original base data.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:152</code></td>
    </tr>
    <tr>
      <td><code>apply_brush_stroke(center: Vector3, radius: float, color: Color, opacity: float, hardness: float)</code></td>
      <td>Paints Gaussians within <code>radius</code> of <code>center</code> with the given color. <code>hardness</code> controls edge falloff (1.0 = hard edge, 0.0 = soft).</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:153</code></td>
    </tr>
    <tr>
      <td><code>get_brush_strokes() -> Array</code></td>
      <td>Returns all recorded brush strokes as an <code>Array</code> of <code>Dictionary</code> entries for serialization or undo.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:154</code></td>
    </tr>
    <tr>
      <td><code>clear_brush_strokes()</code></td>
      <td>Clears all recorded brush stroke history.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:155</code></td>
    </tr>
    <tr>
      <td><code>set_brush_strokes(strokes: Array)</code></td>
      <td>Restores brush strokes from a previously saved <code>Array</code> of <code>Dictionary</code> entries.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:156</code></td>
    </tr>
    <tr>
      <td><code>capture_brush_affected_state(center: Vector3, radius: float) -> Dictionary</code></td>
      <td>Captures the current state of Gaussians within brush radius for undo. Returns a <code>Dictionary</code> with keys <code>"indices"</code>, <code>"colors"</code>, <code>"opacities"</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:157</code></td>
    </tr>
    <tr>
      <td><code>restore_brush_stroke(saved_state: Dictionary)</code></td>
      <td>Restores previously captured brush state, enabling undo of paint operations.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:158</code></td>
    </tr>
  </tbody>
</table>

## Examples
```gdscript
extends Node3D

## Load a .ply point cloud and inspect its bounding volume.
var data := GaussianData.new()

func _ready() -> void:
    var err := data.load_from_file("res://splats/garden.ply")
    if err != OK:
        push_error("Failed to load splat data")
        return
    print("Loaded %d splats" % data.get_count())
    print("Bounding box: ", data.get_aabb())
    print("Memory usage: %.2f MB" % (data.get_memory_usage() / (1024.0 * 1024.0)))

    # Build octree for spatial queries
    data.build_octree(8, 64)
    var nearby := data.query_octree(AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)))
    print("Splats near origin: %d" % nearby.size())
```

```gdscript
extends Node3D

## Runtime brush painting with undo support.
var data: GaussianData

func paint_at(world_pos: Vector3, brush_radius: float, paint_color: Color) -> Dictionary:
    # Capture state before painting so we can undo later
    var saved_state := data.capture_brush_affected_state(world_pos, brush_radius)
    data.apply_brush_stroke(world_pos, brush_radius, paint_color, 0.8, 0.6)
    return saved_state

func undo_paint(saved_state: Dictionary) -> void:
    data.restore_brush_stroke(saved_state)

func _input(event: InputEvent) -> void:
    if event is InputEventKey and event.pressed and event.keycode == KEY_ENTER:
        # Make paint permanent
        data.commit_runtime_changes()
    elif event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
        # Discard all unpainted changes
        data.revert_runtime_changes()
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
      <td><code>load_from_file()</code> returns an error.</td>
      <td>Verify the file path exists and is a supported format (<code>.ply</code> or <code>.spz</code>). Check the output log for missing required PLY properties.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:112</code></td>
    </tr>
    <tr>
      <td>Batch setter has no visible effect.</td>
      <td>Call <code>resize(count)</code> before calling <code>set_positions()</code> or other batch setters. The array length must match <code>get_count()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:84</code></td>
    </tr>
    <tr>
      <td><code>query_octree()</code> returns an empty array.</td>
      <td>Call <code>build_octree()</code> after loading or modifying data. The octree is invalidated when storage changes.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:116</code></td>
    </tr>
    <tr>
      <td>Runtime paint strokes disappear after scene reload.</td>
      <td>Call <code>commit_runtime_changes()</code> followed by <code>save_to_file()</code> to persist paint overlays to disk.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:151</code></td>
    </tr>
    <tr>
      <td>Animation methods return base values instead of animated values.</td>
      <td>Confirm that <code>set_animation_state_machine()</code> was called with a valid resource, <code>is_animation_enabled()</code> returns <code>true</code>, and <code>update_animation(delta)</code> is being called each frame.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp:125</code></td>
    </tr>
  </tbody>
</table>
