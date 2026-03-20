# GaussianSplatAsset API Reference

## Purpose
Use `GaussianSplatAsset` to store, serialize, and exchange Gaussian splat data as a Godot `Resource` (`modules/gaussian_splatting/core/gaussian_splat_asset.h:14`). It holds packed buffer arrays for positions, colors, scales, rotations, spherical harmonics, opacity logits, and painterly metadata, and provides file I/O, format conversion to/from `GaussianData`, and import pipeline metadata. The native resource extension is `.gaussiansplat`.

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
      <td>Load asset from a PLY or SPZ file.</td>
      <td><code>load_from_file(path)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:78</code></td>
    </tr>
    <tr>
      <td>Save asset to disk.</td>
      <td><code>save_to_file(path)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:79</code></td>
    </tr>
    <tr>
      <td>Convert to runtime <code>GaussianData</code>.</td>
      <td><code>get_gaussian_data()</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:915</code></td>
    </tr>
    <tr>
      <td>Populate from runtime <code>GaussianData</code>.</td>
      <td><code>populate_from_gaussian_data(data)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:961</code></td>
    </tr>
    <tr>
      <td>Check load status.</td>
      <td><code>is_loaded()</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:17</code></td>
    </tr>
    <tr>
      <td>Read structured data (vectors, quaternions).</td>
      <td><code>get_position_vectors()</code>, <code>get_scale_vectors()</code>, <code>get_rotation_quaternions()</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:24</code></td>
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
      <td><code>AssetType</code></td>
      <td><code>ASSET_TYPE_STATIC</code> (immutable, optimized for GPU), <code>ASSET_TYPE_DYNAMIC</code> (editable, supports runtime modifications)</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.h:19</code></td>
    </tr>
    <tr>
      <td><code>CompressionFlags</code></td>
      <td><code>COMPRESSION_NONE</code>, <code>COMPRESSION_POSITIONS</code>, <code>COMPRESSION_COLORS</code>, <code>COMPRESSION_SCALES</code>, <code>COMPRESSION_ROTATIONS</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.h:24</code></td>
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
      <td><code>asset_type</code></td>
      <td><code>int (AssetType)</code></td>
      <td><code>set_asset_type</code>, <code>get_asset_type</code></td>
      <td>Enum hint: <code>Static,Dynamic</code>. Controls whether the asset is GPU-optimized or supports runtime edits.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:83</code></td>
    </tr>
    <tr>
      <td><code>splat_count</code></td>
      <td><code>int</code></td>
      <td><code>set_splat_count</code>, <code>get_splat_count</code></td>
      <td>Total number of splats. Setting this resizes all internal buffers and invalidates caches.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:84</code></td>
    </tr>
    <tr>
      <td><code>import/quality_preset</code></td>
      <td><code>String</code></td>
      <td><code>set_import_quality_preset</code>, <code>get_import_quality_preset</code></td>
      <td>Enum hint: <code>low,medium,high,ultra,custom</code>. Stored as lowercase.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:85</code></td>
    </tr>
    <tr>
      <td><code>import/compression_flags</code></td>
      <td><code>int (CompressionFlags)</code></td>
      <td><code>set_compression_flags</code>, <code>get_compression_flags</code></td>
      <td>Bitfield hint: <code>Positions,Colors,Scales,Rotations</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:87</code></td>
    </tr>
    <tr>
      <td><code>import/metadata</code></td>
      <td><code>Dictionary</code></td>
      <td><code>set_import_metadata</code>, <code>get_import_metadata</code></td>
      <td>Stores import pipeline metadata. Automatically includes splat count, quality preset, and compression flags.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:89</code></td>
    </tr>
    <tr>
      <td><code>import/thumbnail</code></td>
      <td><code>Texture2D</code></td>
      <td><code>set_thumbnail</code>, <code>get_thumbnail</code></td>
      <td>Optional preview thumbnail for the asset browser.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:90</code></td>
    </tr>
    <tr>
      <td><code>import/source_path</code></td>
      <td><code>String</code></td>
      <td><code>set_source_path</code>, <code>get_source_path</code></td>
      <td>File hint: <code>*.ply,*.spz</code>. The original file path used during import.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:92</code></td>
    </tr>
    <tr>
      <td><code>data/positions</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_positions</code>, <code>get_positions</code></td>
      <td>Packed x,y,z floats (3 per splat). Setting this updates <code>splat_count</code> from array size.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:94</code></td>
    </tr>
    <tr>
      <td><code>data/colors</code></td>
      <td><code>PackedColorArray</code></td>
      <td><code>set_colors</code>, <code>get_colors</code></td>
      <td>RGBA color per splat.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:95</code></td>
    </tr>
    <tr>
      <td><code>data/scales</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_scales</code>, <code>get_scales</code></td>
      <td>Per-axis scale (3 floats per splat). New entries default to <code>(1, 1, 1)</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:96</code></td>
    </tr>
    <tr>
      <td><code>data/rotations</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_rotations</code>, <code>get_rotations</code></td>
      <td>Quaternion components (4 floats per splat, w,x,y,z order). New entries default to identity.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:97</code></td>
    </tr>
    <tr>
      <td><code>data/sh_dc</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_sh_dc_coefficients</code>, <code>get_sh_dc_coefficients</code></td>
      <td>Spherical harmonics DC band (3 RGB floats per splat).</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:98</code></td>
    </tr>
    <tr>
      <td><code>data/sh_first_order</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_sh_first_order_coefficients</code>, <code>get_sh_first_order_coefficients</code></td>
      <td>First-order SH coefficients. Term count auto-calculated from array size and <code>splat_count</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:99</code></td>
    </tr>
    <tr>
      <td><code>data/sh_high_order</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_sh_high_order_coefficients</code>, <code>get_sh_high_order_coefficients</code></td>
      <td>Higher-order SH coefficients. Term count auto-calculated from array size and <code>splat_count</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:100</code></td>
    </tr>
    <tr>
      <td><code>data/opacity_logits</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_opacity_logits</code>, <code>get_opacity_logits</code></td>
      <td>Raw opacity logits per splat (sigmoid-encoded). Use <code>get_opacities()</code> for decoded [0, 1] values.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:101</code></td>
    </tr>
    <tr>
      <td><code>data/palette_ids</code></td>
      <td><code>PackedInt32Array</code></td>
      <td><code>set_palette_ids</code>, <code>get_palette_ids</code></td>
      <td>Palette lookup indices per splat for painterly color mapping.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:102</code></td>
    </tr>
    <tr>
      <td><code>data/painterly_flags</code></td>
      <td><code>PackedInt32Array</code></td>
      <td><code>set_painterly_flags</code>, <code>get_painterly_flags</code></td>
      <td>Shared storage for painterly flags and brush override IDs per splat.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:103</code></td>
    </tr>
    <tr>
      <td><code>data/normals</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_normals</code>, <code>get_normals</code></td>
      <td>Optional per-splat normals (3 floats per splat) for surfel mode.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:104</code></td>
    </tr>
    <tr>
      <td><code>data/brush_axes</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_brush_axes</code>, <code>get_brush_axes</code></td>
      <td>Painterly brush axis vectors (2 floats per splat).</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:105</code></td>
    </tr>
    <tr>
      <td><code>data/stroke_ages</code></td>
      <td><code>PackedFloat32Array</code></td>
      <td><code>set_stroke_ages</code>, <code>get_stroke_ages</code></td>
      <td>Painterly stroke age metadata per splat.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:106</code></td>
    </tr>
  </tbody>
</table>

### Methods

#### Status and Conversion
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
      <td><code>is_loaded() -> bool</code></td>
      <td>Returns <code>true</code> when the asset has been populated with splat data (<code>splat_count > 0</code>).</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:17</code></td>
    </tr>
    <tr>
      <td><code>get_instance_count() -> int</code> (static)</td>
      <td>Returns the total number of live <code>GaussianSplatAsset</code> instances. Useful for memory tracking.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:81</code></td>
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
      <td>Loads Gaussian splat data from a <code>.ply</code> or <code>.spz</code> file. Populates all internal buffers via the data loader and <code>populate_from_gaussian_data()</code>. Returns <code>OK</code> on success, <code>ERR_FILE_NOT_FOUND</code> if the file is missing, or <code>ERR_FILE_CORRUPT</code> if required PLY properties are absent.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:863</code></td>
    </tr>
    <tr>
      <td><code>save_to_file(path: String) -> Error</code></td>
      <td>Converts the asset to a <code>GaussianData</code> instance and saves it to a PLY file. Returns <code>ERR_INVALID_DATA</code> if the conversion fails.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:907</code></td>
    </tr>
  </tbody>
</table>

#### Structured Data Getters
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
      <td><code>get_position_vectors() -> PackedVector3Array</code></td>
      <td>Converts packed position floats into a <code>PackedVector3Array</code>. Missing entries are filled with <code>Vector3.ZERO</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:262</code></td>
    </tr>
    <tr>
      <td><code>get_scale_vectors() -> PackedVector3Array</code></td>
      <td>Converts packed scale floats into a <code>PackedVector3Array</code>. Missing entries default to <code>Vector3(1, 1, 1)</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:285</code></td>
    </tr>
    <tr>
      <td><code>get_rotation_quaternions() -> TypedArray[Quaternion]</code></td>
      <td>Converts packed rotation floats (w,x,y,z) into a <code>TypedArray[Quaternion]</code>. Zero-length quaternions default to identity.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:308</code></td>
    </tr>
    <tr>
      <td><code>get_spherical_harmonics_buffer() -> PackedFloat32Array</code></td>
      <td>Assembles a combined SH buffer (DC + first-order + high-order) for all splats. Falls back to color data for DC when SH DC coefficients are absent.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:338</code></td>
    </tr>
    <tr>
      <td><code>get_opacities() -> PackedFloat32Array</code></td>
      <td>Decodes opacity logits through a sigmoid function into [0, 1] values. Falls back to <code>color.a</code> when logits are absent.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:418</code></td>
    </tr>
    <tr>
      <td><code>get_palette_ids_buffer() -> PackedInt32Array</code></td>
      <td>Returns palette IDs clamped to [0, 65535], padded with zeros for missing entries.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:448</code></td>
    </tr>
    <tr>
      <td><code>get_painterly_flags_buffer() -> PackedInt32Array</code></td>
      <td>Returns painterly flags clamped to [0, 65535], padded with zeros for missing entries.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:467</code></td>
    </tr>
    <tr>
      <td><code>get_brush_override_ids() -> PackedInt32Array</code></td>
      <td>Returns brush override IDs from the shared painterly flags storage lane.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:228</code></td>
    </tr>
    <tr>
      <td><code>get_brush_override_ids_buffer() -> PackedInt32Array</code></td>
      <td>Returns sanitized brush override IDs. Delegates to <code>get_painterly_flags_buffer()</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:486</code></td>
    </tr>
    <tr>
      <td><code>get_normal_vectors() -> PackedVector3Array</code></td>
      <td>Converts packed normal floats into a <code>PackedVector3Array</code>. Missing entries default to <code>Vector3(0, 1, 0)</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:490</code></td>
    </tr>
    <tr>
      <td><code>get_brush_axes_vector2() -> PackedVector2Array</code></td>
      <td>Converts packed brush axis floats into a <code>PackedVector2Array</code>. Missing entries default to <code>Vector2(1, 1)</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:513</code></td>
    </tr>
    <tr>
      <td><code>get_stroke_ages_buffer() -> PackedFloat32Array</code></td>
      <td>Returns stroke ages padded with <code>0.0</code> for missing entries.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:536</code></td>
    </tr>
    <tr>
      <td><code>get_sh_first_order_terms() -> int</code></td>
      <td>Returns the number of first-order SH coefficient vectors per splat (0-3).</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:48</code></td>
    </tr>
    <tr>
      <td><code>get_sh_high_order_terms() -> int</code></td>
      <td>Returns the number of higher-order SH coefficient vectors per splat.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:49</code></td>
    </tr>
    <tr>
      <td><code>set_sh_component_terms(first_order_terms: int, high_order_terms: int)</code></td>
      <td>Directly sets the SH term counts. <code>first_order_terms</code> is clamped to a maximum of 3.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:66</code></td>
    </tr>
  </tbody>
</table>

## Examples
```gdscript
extends Node3D

## Load a .ply file into a GaussianSplatAsset and inspect metadata.
func _ready() -> void:
    var asset := GaussianSplatAsset.new()
    var err := asset.load_from_file("res://splats/scene.ply")
    if err != OK:
        push_error("Failed to load asset")
        return

    print("Splat count: %d" % asset.get_splat_count())
    print("Asset type: %d" % asset.get_asset_type())
    print("Quality preset: %s" % asset.get_import_quality_preset())
    print("Has SH: first=%d high=%d" % [asset.get_sh_first_order_terms(), asset.get_sh_high_order_terms()])

    # Convert to GaussianData for spatial queries and painting
    var data := asset.get_gaussian_data()
    if data != null:
        print("AABB: ", data.get_aabb())
```

```gdscript
extends Node3D

## Build a GaussianSplatAsset from procedural data and save it.
func create_procedural_asset() -> GaussianSplatAsset:
    var asset := GaussianSplatAsset.new()
    asset.set_asset_type(GaussianSplatAsset.ASSET_TYPE_DYNAMIC)

    var count := 1000
    var positions := PackedFloat32Array()
    positions.resize(count * 3)
    var colors := PackedColorArray()
    colors.resize(count)

    for i in range(count):
        var angle := float(i) / float(count) * TAU
        positions[i * 3 + 0] = cos(angle) * 2.0
        positions[i * 3 + 1] = sin(float(i) * 0.1) * 0.5
        positions[i * 3 + 2] = sin(angle) * 2.0
        colors[i] = Color.from_hsv(float(i) / float(count), 0.8, 1.0)

    asset.set_positions(positions)
    asset.set_colors(colors)
    asset.set_import_quality_preset("high")
    asset.set_compression_flags(
        GaussianSplatAsset.COMPRESSION_POSITIONS | GaussianSplatAsset.COMPRESSION_COLORS
    )

    var save_err := asset.save_to_file("res://splats/procedural.ply")
    if save_err != OK:
        push_error("Failed to save procedural asset")
    return asset
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
      <td><code>is_loaded()</code> returns <code>false</code> after calling setters.</td>
      <td>Ensure <code>set_positions()</code> was called with a non-empty array. The <code>splat_count</code> is derived from the positions array (3 floats per splat).</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:554</code></td>
    </tr>
    <tr>
      <td>Getters log "called on unloaded asset" warnings.</td>
      <td>Check that <code>load_from_file()</code> returned <code>OK</code> or that <code>splat_count</code> was set before reading data. All raw-array getters warn once when both <code>splat_count</code> and the array are empty.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:158</code></td>
    </tr>
    <tr>
      <td><code>get_opacities()</code> returns unexpected values.</td>
      <td>Opacity logits are stored in sigmoid space. If you set raw logit values, the decoded opacities follow <code>1 / (1 + exp(-logit))</code>. Use <code>set_colors()</code> with alpha for linear opacity instead.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:418</code></td>
    </tr>
    <tr>
      <td><code>save_to_file()</code> returns <code>ERR_INVALID_DATA</code>.</td>
      <td>The asset must have valid data (<code>splat_count > 0</code>) for conversion to <code>GaussianData</code> to succeed before saving. Verify that <code>is_loaded()</code> returns <code>true</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:907</code></td>
    </tr>
    <tr>
      <td>SH term counts are incorrect after setting coefficient arrays.</td>
      <td>Term counts are auto-calculated from <code>array.size() / (splat_count * 3)</code>. Set <code>splat_count</code> (via <code>set_positions()</code> or <code>set_splat_count()</code>) before setting SH arrays, or use <code>set_sh_component_terms()</code> to override.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp:618</code></td>
    </tr>
  </tbody>
</table>
