class_name BenchmarkSceneContract
extends RefCounted

const MANIFEST_PATH := "res://tests/fixtures/benchmark_asset_manifest.json"
const CHUNKED_LADDER_REF_PREFIX := "chunked_ladder:"
const RUNNABLE_CHUNKED_STAGING_STATUS := "materialized"
const ALLOWED_CHUNKED_WORLD_CONTRACT_LANE_IDS := ["open_world_corridor_proof"]

static var _manifest_loaded := false
static var _manifest_cache: Dictionary = {}

static func scene_id_from_path(scene_file_path: String) -> String:
	if scene_file_path.is_empty():
		return "unknown_scene"
	var scene_id := scene_file_path.get_file().get_basename().strip_edges().to_lower()
	if scene_id.begins_with("lane_"):
		return "benchmark_suite_lane"
	return scene_id

static func resolve_contract(scene_id: String, defaults: Dictionary, pending_contract: Dictionary) -> Dictionary:
	var merged := defaults.duplicate(true)
	var cli_contract := _parse_cmdline_contract()
	if not cli_contract.is_empty():
		merged.merge(cli_contract, true)
	if not pending_contract.is_empty():
		merged.merge(pending_contract, true)

	merged["scene_id"] = scene_id
	if str(merged.get("lane_id", "")).is_empty():
		merged["lane_id"] = scene_id
	return merged

static func resolve_asset_path(
	scene_id: String,
	lane_id: String,
	asset_override_path: String,
	placeholder_asset_path: String
) -> String:
	if not asset_override_path.is_empty():
		return asset_override_path
	if not placeholder_asset_path.is_empty():
		return placeholder_asset_path

	var manifest := _load_manifest()
	var lane_defaults = manifest.get("lane_defaults", {})
	if lane_defaults is Dictionary and lane_defaults.has(lane_id):
		return _resolve_manifest_asset_path(manifest, str(lane_defaults.get(lane_id, "")))

	var scene_defaults = manifest.get("scene_defaults", {})
	if scene_defaults is Dictionary and scene_defaults.has(scene_id):
		return _resolve_manifest_asset_path(manifest, str(scene_defaults.get(scene_id, "")))

	return _resolve_manifest_asset_path(manifest, str(manifest.get("default_asset", "")))

static func resolve_world_contract_path(
	scene_id: String,
	lane_id: String,
	asset_override_path: String
) -> String:
	if not asset_override_path.is_empty():
		return asset_override_path
	if not ALLOWED_CHUNKED_WORLD_CONTRACT_LANE_IDS.has(lane_id):
		return ""

	var manifest := _load_manifest()
	var lane_defaults = manifest.get("lane_defaults", {})
	if lane_defaults is Dictionary and lane_defaults.has(lane_id):
		return _resolve_manifest_world_contract_path(manifest, str(lane_defaults.get(lane_id, "")))

	var scene_defaults = manifest.get("scene_defaults", {})
	if scene_defaults is Dictionary and scene_defaults.has(scene_id):
		return _resolve_manifest_world_contract_path(manifest, str(scene_defaults.get(scene_id, "")))

	return _resolve_manifest_world_contract_path(manifest, str(manifest.get("default_asset", "")))

static func _load_manifest() -> Dictionary:
	if _manifest_loaded:
		return _manifest_cache
	_manifest_loaded = true
	_manifest_cache = {}

	if not FileAccess.file_exists(MANIFEST_PATH):
		return _manifest_cache

	var handle := FileAccess.open(MANIFEST_PATH, FileAccess.READ)
	if handle == null:
		return _manifest_cache

	var parsed = JSON.parse_string(handle.get_as_text())
	if parsed is Dictionary:
		_manifest_cache = parsed
	return _manifest_cache

static func _resolve_manifest_asset_path(manifest: Dictionary, raw_value: String) -> String:
	if raw_value.is_empty():
		return ""
	if not raw_value.begins_with(CHUNKED_LADDER_REF_PREFIX):
		return raw_value

	var asset_id := raw_value.trim_prefix(CHUNKED_LADDER_REF_PREFIX)
	var ladder = manifest.get("chunked_asset_ladder", {})
	if not (ladder is Dictionary) or not ladder.has(asset_id):
		return raw_value
	var entry = ladder.get(asset_id, {})
	if not (entry is Dictionary):
		return raw_value
	if str(entry.get("staging_status", "")) != RUNNABLE_CHUNKED_STAGING_STATUS:
		return raw_value
	var staging = entry.get("staging", {})
	if not (staging is Dictionary):
		return raw_value
	var staged_path := str(staging.get("project_benchmark_asset_path", ""))
	return staged_path if not staged_path.is_empty() else raw_value

static func _resolve_manifest_world_contract_path(manifest: Dictionary, raw_value: String) -> String:
	if raw_value.is_empty():
		return ""
	if not raw_value.begins_with(CHUNKED_LADDER_REF_PREFIX):
		return raw_value

	var asset_id := raw_value.trim_prefix(CHUNKED_LADDER_REF_PREFIX)
	var ladder = manifest.get("chunked_asset_ladder", {})
	if not (ladder is Dictionary) or not ladder.has(asset_id):
		return raw_value
	var entry = ladder.get(asset_id, {})
	if not (entry is Dictionary):
		return raw_value
	var staging = entry.get("staging", {})
	if not (staging is Dictionary):
		return raw_value
	# Prefer pre-built staged world when the entry is materialized.
	if str(entry.get("staging_status", "")) == RUNNABLE_CHUNKED_STAGING_STATUS:
		var staged_world := str(staging.get("project_staged_world_path", ""))
		if not staged_world.is_empty():
			return staged_world
	# Fall back to the bootstrap stage-manifest path for runtime synthesis.
	var builder = entry.get("bootstrap_world_builder", {})
	if not (builder is Dictionary) or builder.is_empty():
		return raw_value
	var staged_path := str(staging.get("project_stage_manifest_path", ""))
	return staged_path if not staged_path.is_empty() else raw_value

static func _parse_cmdline_contract() -> Dictionary:
	var out := {}
	var args := OS.get_cmdline_args()
	var i := 0
	while i < args.size():
		var arg := str(args[i])
		match arg:
			"--benchmark-headless-summary":
				out["headless_summary"] = true
			"--benchmark-orchestrated":
				out["orchestrated"] = true
			"--benchmark-duration", "--duration":
				i += 1
				if i < args.size():
					out["duration_s"] = float(args[i])
			"--benchmark-warmup":
				i += 1
				if i < args.size():
					out["warmup_s"] = float(args[i])
			"--benchmark-output":
				i += 1
				if i < args.size():
					out["output_path"] = str(args[i])
			"--benchmark-asset", "--ply-path":
				i += 1
				if i < args.size():
					out["asset_path"] = str(args[i])
			"--benchmark-lane-tag":
				i += 1
				if i < args.size():
					out["lane_tag"] = str(args[i])
			"--benchmark-capture-dir":
				i += 1
				if i < args.size():
					out["capture_dir"] = str(args[i])
			"--benchmark-reference-dir":
				i += 1
				if i < args.size():
					out["reference_dir"] = str(args[i])
			"--benchmark-capture-tag":
				i += 1
				if i < args.size():
					out["capture_tag"] = str(args[i])
			"--benchmark-instancing-mode":
				i += 1
				if i < args.size():
					out["instancing_mode"] = str(args[i])
			"--benchmark-lane-id":
				i += 1
				if i < args.size():
					out["lane_id"] = str(args[i])
			"--benchmark-lane-name":
				i += 1
				if i < args.size():
					out["lane_name"] = str(args[i])
			"--benchmark-lane-description":
				i += 1
				if i < args.size():
					out["lane_description"] = str(args[i])
			"--benchmark-lane-preset":
				i += 1
				if i < args.size():
					out["lane_preset"] = str(args[i])
			"--benchmark-camera-mode", "--camera-mode":
				i += 1
				if i < args.size():
					out["camera_mode"] = str(args[i])
			"--benchmark-ssim-threshold":
				i += 1
				if i < args.size():
					out["ssim_threshold"] = float(args[i])
			"--benchmark-psnr-threshold":
				i += 1
				if i < args.size():
					out["psnr_threshold"] = float(args[i])
			_:
				if arg.begins_with("--benchmark-duration="):
					out["duration_s"] = float(arg.substr(len("--benchmark-duration=")))
				elif arg.begins_with("--duration="):
					out["duration_s"] = float(arg.substr(len("--duration=")))
				elif arg.begins_with("--benchmark-warmup="):
					out["warmup_s"] = float(arg.substr(len("--benchmark-warmup=")))
				elif arg.begins_with("--benchmark-output="):
					out["output_path"] = arg.substr(len("--benchmark-output="))
				elif arg.begins_with("--benchmark-asset="):
					out["asset_path"] = arg.substr(len("--benchmark-asset="))
				elif arg.begins_with("--ply-path="):
					out["asset_path"] = arg.substr(len("--ply-path="))
				elif arg.begins_with("--benchmark-lane-tag="):
					out["lane_tag"] = arg.substr(len("--benchmark-lane-tag="))
				elif arg.begins_with("--benchmark-capture-dir="):
					out["capture_dir"] = arg.substr(len("--benchmark-capture-dir="))
				elif arg.begins_with("--benchmark-reference-dir="):
					out["reference_dir"] = arg.substr(len("--benchmark-reference-dir="))
				elif arg.begins_with("--benchmark-capture-tag="):
					out["capture_tag"] = arg.substr(len("--benchmark-capture-tag="))
				elif arg.begins_with("--benchmark-instancing-mode="):
					out["instancing_mode"] = arg.substr(len("--benchmark-instancing-mode="))
				elif arg.begins_with("--benchmark-camera-mode="):
					out["camera_mode"] = arg.substr(len("--benchmark-camera-mode="))
				elif arg.begins_with("--camera-mode="):
					out["camera_mode"] = arg.substr(len("--camera-mode="))
				elif arg.begins_with("--benchmark-ssim-threshold="):
					out["ssim_threshold"] = float(arg.substr(len("--benchmark-ssim-threshold=")))
				elif arg.begins_with("--benchmark-psnr-threshold="):
					out["psnr_threshold"] = float(arg.substr(len("--benchmark-psnr-threshold=")))
		i += 1

	return out
