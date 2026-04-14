class_name BenchmarkOpenWorldStageContract
extends RefCounted

static func load_stage_manifest(path: String) -> Dictionary:
	if path.is_empty() or not FileAccess.file_exists(path):
		return {}

	var handle := FileAccess.open(path, FileAccess.READ)
	if handle == null:
		return {}

	var parsed = JSON.parse_string(handle.get_as_text())
	return parsed if parsed is Dictionary else {}


static func build_world_from_stage_manifest(stage_manifest_path: String, owner: Node) -> Dictionary:
	var stage_manifest := load_stage_manifest(stage_manifest_path)
	if stage_manifest.is_empty():
		return {"error": "stage_manifest_unavailable"}

	var builder = stage_manifest.get("bootstrap_world_builder", {})
	if not (builder is Dictionary) or builder.is_empty():
		return {"error": "bootstrap_world_builder_missing", "stage_manifest": stage_manifest}

	var source_asset_path := str(builder.get("source_asset_path", ""))
	if source_asset_path.is_empty():
		return {"error": "source_asset_path_missing", "stage_manifest": stage_manifest}

	var source_asset := GaussianSplatAsset.new()
	var load_err := source_asset.load_from_file(source_asset_path)
	if load_err != OK:
		return {
			"error": "source_asset_load_failed",
			"source_asset_path": source_asset_path,
			"load_error": load_err,
			"stage_manifest": stage_manifest,
		}

	var container := GaussianSplatContainer.new()
	container.name = "OpenWorldBootstrapBuilder"
	container.set_merge_on_ready(false)
	container.set_chunk_size(float(builder.get("chunk_size", 0.75)))
	owner.add_child(container)

	var corridor_lanes: int = max(1, int(builder.get("corridor_lanes", 4)))
	var corridor_segments: int = max(1, int(builder.get("corridor_segments", 32)))
	var instance_count: int = max(1, int(builder.get("instance_count", corridor_lanes * corridor_segments)))
	var lane_spacing := float(builder.get("lane_spacing", 7.5))
	var segment_spacing := float(builder.get("segment_spacing", 10.0))
	var vertical_wave := float(builder.get("vertical_wave_amplitude", 0.4))
	var lateral_wave := float(builder.get("lateral_wave_amplitude", 1.2))

	for instance_index: int in range(instance_count):
		var lane: int = instance_index % corridor_lanes
		var segment: int = instance_index / corridor_lanes
		var node := GaussianSplatNode3D.new()
		node.name = "Bootstrap_%04d_%02d" % [segment, lane]
		node.set_splat_asset(source_asset)
		var lane_offset := (float(lane) - float(corridor_lanes - 1) * 0.5) * lane_spacing
		var segment_offset := -float(segment) * segment_spacing
		node.position = Vector3(
			lane_offset + sin(float(segment) * 0.19) * lateral_wave,
			cos(float(segment) * 0.11 + float(lane) * 0.7) * vertical_wave,
			segment_offset
		)
		container.add_child(node)

	container.merge_children()
	var generated_chunk_count := int(container.get_chunk_count())
	var world := container.export_world_resource()
	container.queue_free()
	if world == null:
		return {"error": "world_export_failed", "stage_manifest": stage_manifest}

	var metadata := world.get_metadata()
	if not (metadata is Dictionary):
		metadata = {}
	var working_set: Dictionary = stage_manifest.get("working_set_contract", {})
	metadata["benchmark_stage_manifest_path"] = stage_manifest_path
	metadata["open_world_asset_id"] = str(stage_manifest.get("asset_id", ""))
	metadata["open_world_contract_total_splats"] = int(working_set.get("total_splats", 0))
	metadata["open_world_materialized_total_splats"] = int(builder.get("materialized_total_splats", 0))
	metadata["open_world_materialized_instance_count"] = instance_count
	metadata["open_world_materialized_chunk_count"] = generated_chunk_count
	world.set_metadata(metadata)

	return {
		"world": world,
		"stage_manifest": stage_manifest,
		"generated_chunk_count": generated_chunk_count,
		"materialized_total_splats": int(builder.get("materialized_total_splats", 0)),
		"materialized_instance_count": instance_count,
		"contract_total_splats": int(working_set.get("total_splats", 0)),
		"asset_id": str(stage_manifest.get("asset_id", "")),
	}
