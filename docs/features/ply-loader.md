# PLY Loader

!!! info "Scope"
    For programmers and technical users who need loader behavior, property validation, or runtime save/load details.
    This page covers the technical loader layer, not the end-user import flow.
    It complements the canonical [Gaussian Splat Asset Import Workflow](../workflows/importing.md).

## Purpose

Load and inspect Gaussian PLY data with `PLYLoader`, or load and save PLY through `GaussianData`.

## Usage

| Step | Action | Implementation reference |
| --- | --- | --- |
| 1 | Use `GaussianData.load_from_file(path)` for runtime loading. | `modules/gaussian_splatting/core/gaussian_data.cpp:174`, `modules/gaussian_splatting/core/gaussian_data.cpp:969` |
| 2 | Use `PLYLoader.load_file(path)` when you need property inspection or load statistics. | `modules/gaussian_splatting/io/ply_loader.cpp:61`, `modules/gaussian_splatting/io/ply_loader.cpp:69` |
| 3 | Use `GaussianData.save_to_file(path)` to export binary little-endian PLY. | `modules/gaussian_splatting/core/gaussian_data.cpp:175`, `modules/gaussian_splatting/core/gaussian_data.cpp:1064`, `modules/gaussian_splatting/core/gaussian_data.cpp:1074` |
| 4 | In editor import flow, use importer `gaussian_splat_ply` which outputs `GaussianSplatAsset` (`.tres`). | `modules/gaussian_splatting/io/resource_importer_ply.cpp:137`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:153`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:149` |

## API

| API | Type | Behavior | Implementation reference |
| --- | --- | --- | --- |
| `GaussianData.load_from_file(path)` | Runtime method | Routes by extension and loads PLY through `PLYLoader` unless extension is `.spz`. | `modules/gaussian_splatting/io/gaussian_data_loader.cpp:9`, `modules/gaussian_splatting/io/gaussian_data_loader.cpp:29` |
| `GaussianData.save_to_file(path)` | Runtime method | Writes PLY with position, SH DC, scale, rotation, opacity, and painterly fields. | `modules/gaussian_splatting/core/gaussian_data.cpp:1077`, `modules/gaussian_splatting/core/gaussian_data.cpp:1089`, `modules/gaussian_splatting/core/gaussian_data.cpp:1108` |
| `PLYLoader.load_file(path)` | Runtime method | Parses header, optionally uses `.gsplatworld` cache, then parses binary or ASCII vertex data. | `modules/gaussian_splatting/io/ply_loader.cpp:69`, `modules/gaussian_splatting/io/ply_loader.cpp:97`, `modules/gaussian_splatting/io/ply_loader.cpp:112` |
| `PLYLoader.get_property_deficiencies()` | Runtime method | Returns missing required and optional PLY properties. | `modules/gaussian_splatting/io/ply_loader.cpp:907` |
| `PLYLoader.get_load_statistics()` | Runtime method | Returns count, format, property count, timings, cache-hit flag, and bounds. | `modules/gaussian_splatting/io/ply_loader.cpp:885` |

| PLY property set | Required | Notes | Implementation reference |
| --- | --- | --- | --- |
| `x,y,z` | Yes | Position fields are required by validation and deficiency checks. | `modules/gaussian_splatting/io/ply_loader.cpp:911`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:482` |
| `f_dc_0,f_dc_1,f_dc_2` | Yes | Color is read as SH DC and converted using `SH_C0`. | `modules/gaussian_splatting/io/ply_loader.cpp:336`, `modules/gaussian_splatting/io/ply_loader.cpp:692` |
| `scale_0,scale_1,scale_2` | Yes | Scale is decoded with `exp`. | `modules/gaussian_splatting/io/ply_loader.cpp:413` |
| `rot_0..rot_3` | Yes | Rotation is read as quaternion and normalized. | `modules/gaussian_splatting/io/ply_loader.cpp:419`, `modules/gaussian_splatting/io/ply_loader.cpp:424` |
| `opacity` | Yes | Opacity is decoded as sigmoid from logit. | `modules/gaussian_splatting/io/ply_loader.cpp:427` |
| `nx,ny,nz` | No | If present, loader enables 2D mode. | `modules/gaussian_splatting/io/ply_loader.cpp:323`, `modules/gaussian_splatting/io/ply_loader.cpp:327` |
| `palette_id,brush_axis_u,brush_axis_v,stroke_age` | No | Painterly metadata is loaded when present and written on save. | `modules/gaussian_splatting/io/ply_loader.cpp:331`, `modules/gaussian_splatting/core/gaussian_data.cpp:1108` |
| `f_rest_*` | No | Higher-order SH is repacked from channel-major to coefficient-major RGB. | `modules/gaussian_splatting/io/ply_loader.cpp:703`, `modules/gaussian_splatting/io/ply_loader.cpp:785` |

| Format path | Read support | Write support | Implementation reference |
| --- | --- | --- | --- |
| `ascii` | Yes | No | `modules/gaussian_splatting/io/ply_loader.cpp:158`, `modules/gaussian_splatting/io/ply_loader.cpp:517` |
| `binary_little_endian` | Yes | Yes | `modules/gaussian_splatting/io/ply_loader.cpp:152`, `modules/gaussian_splatting/core/gaussian_data.cpp:1074` |
| `binary_big_endian` | Yes | No | `modules/gaussian_splatting/io/ply_loader.cpp:155`, `modules/gaussian_splatting/io/ply_loader.cpp:807` |

| Import option | Default | Effect | Implementation reference |
| --- | --- | --- | --- |
| `quality/preset` | preset-specific | Chooses preset baseline (`mobile` to `custom`). | `modules/gaussian_splatting/io/resource_importer_ply.cpp:175` |
| `quality/max_splats` | preset-specific | Caps final splat count after import processing. | `modules/gaussian_splatting/io/resource_importer_ply.cpp:185`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:308` |
| `quality/density_multiplier` | preset-specific | Reduces density and can merge source ranges. | `modules/gaussian_splatting/io/resource_importer_ply.cpp:189`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:309` |
| `validation/validate_required_properties` | `true` | Fails import if required properties are missing or invalid. | `modules/gaussian_splatting/io/resource_importer_ply.cpp:196`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:476` |
| `validation/warn_missing_optional` | `true` | Logs optional property presence and omissions. | `modules/gaussian_splatting/io/resource_importer_ply.cpp:198`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:539` |
| `preview/generate_thumbnail` | `true` | Generates and stores thumbnail in imported asset metadata. | `modules/gaussian_splatting/io/resource_importer_ply.cpp:214`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:381` |

## Examples

```gdscript
extends Node

func load_and_save_ply() -> void:
	var data := GaussianData.new()
	var err := data.load_from_file("res://models/scan.ply")
	if err != OK:
		push_error("load_from_file failed: %d" % err)
		return

	print("Loaded splats:", data.get_count())
	print("Bounds:", data.get_aabb())

	err = data.save_to_file("user://scan_out.ply")
	if err != OK:
		push_error("save_to_file failed: %d" % err)
```

```gdscript
extends Node

func inspect_ply_header() -> void:
	var loader := PLYLoader.new()
	var err := loader.load_file("res://models/scan.ply")
	if err != OK:
		push_error("PLYLoader.load_file failed: %d" % err)
		return

	var stats := loader.get_load_statistics()
	var summary := loader.get_property_summary()

	print("Stats:", stats)
	print("Missing required:", summary.get("missing_required", PackedStringArray()))
	print("Missing optional:", summary.get("missing_optional", PackedStringArray()))
```

## Troubleshooting

| Symptom | Cause | Fix | Implementation reference |
| --- | --- | --- | --- |
| `ERR_FILE_CORRUPT` on load | Required properties are missing in source PLY. | Ensure `x,y,z`, `f_dc_0..2`, `scale_0..2`, `rot_0..3`, and `opacity` exist. | `modules/gaussian_splatting/io/ply_loader.cpp:911`, `modules/gaussian_splatting/core/gaussian_data.cpp:977` |
| Import fails during validation | Validation found missing fields or invalid values. | Re-export with required fields and finite position/scale/opacity values. | `modules/gaussian_splatting/io/resource_importer_ply.cpp:482`, `modules/gaussian_splatting/io/resource_importer_ply.cpp:531` |
| 2D surfel mode is not enabled | Source PLY does not include full normal triplet. | Export `nx,ny,nz` for each vertex. | `modules/gaussian_splatting/io/ply_loader.cpp:521`, `modules/gaussian_splatting/io/ply_loader.cpp:526` |
| Repeated loads are slower than expected | Cache path is disabled or cache metadata mismatch prevents reuse. | Enable `rendering/gaussian_splatting/import/use_gsplatworld_cache` and keep source timestamp and size stable. | `modules/gaussian_splatting/io/ply_loader.cpp:46`, `modules/gaussian_splatting/io/ply_loader.cpp:242` |
