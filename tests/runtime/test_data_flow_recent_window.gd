extends SceneTree

const SKIP_MARKER := "[RUNTIME_SKIP]"
const FAIL_MARKER := "[RUNTIME_FAIL]"
const METRICS_MARKER := "[RUNTIME_METRICS]"

const ASSET_PATH := "res://tests/fixtures/test_splats.ply"
const MAX_TEST_FRAMES := 220
const REQUIRED_SAMPLES := 30

var scene_root: Node3D
var splat_node: GaussianSplatNode3D
var camera: Camera3D
var renderer = null

var metrics: Dictionary = {
	"frames": 0,
	"recent_window_seen": false,
	"recent_frames_recorded_max": 0,
	"recent_capacity": 0,
	"recent_frame_deltas_size_max": 0,
	"lifetime_pack_sh_samples": 0,
	"lifetime_pack_range_calls": 0,
	"recent_pack_sh_samples": 0,
	"recent_pack_range_calls": 0,
	"status": "",
	"reason": "",
}

func _init() -> void:
	call_deferred("_run")

func _is_headless_runtime() -> bool:
	return OS.has_feature("headless") or DisplayServer.get_name() == "headless"

func _record_failure(reason: String) -> void:
	push_error("%s %s" % [FAIL_MARKER, reason])

func _emit_metrics(status: String, reason: String) -> void:
	metrics["status"] = status
	metrics["reason"] = reason
	print("%s %s" % [METRICS_MARKER, JSON.stringify(metrics)])

func _cleanup() -> void:
	if scene_root != null:
		scene_root.queue_free()
	scene_root = null
	splat_node = null
	camera = null
	renderer = null

func _setup_scene() -> bool:
	scene_root = Node3D.new()
	scene_root.name = "DataFlowRecentWindowRoot"
	get_root().add_child(scene_root)

	camera = Camera3D.new()
	camera.name = "DataFlowCamera"
	camera.position = Vector3(0.0, 1.8, 6.0)
	camera.look_at(Vector3.ZERO, Vector3.UP)
	camera.make_current()
	scene_root.add_child(camera)

	splat_node = GaussianSplatNode3D.new()
	splat_node.name = "DataFlowSplat"
	scene_root.add_child(splat_node)

	var asset := GaussianSplatAsset.new()
	var load_err := asset.load_from_file(ASSET_PATH)
	if load_err != OK:
		_record_failure("Failed to load fixture asset %s (err=%d)" % [ASSET_PATH, load_err])
		return false
	splat_node.set_splat_asset(asset)
	return true

func _sum_delta_field(deltas: Array, field_name: String) -> int:
	var total := 0
	for entry in deltas:
		if entry is Dictionary:
			total += int(entry.get(field_name, 0))
	return total

func _run() -> void:
	if _is_headless_runtime():
		var skip_reason := "Data flow recent window requires non-headless execution."
		_emit_metrics("skipped", skip_reason)
		print("%s %s" % [SKIP_MARKER, skip_reason])
		quit(0)
		return

	if not _setup_scene():
		_emit_metrics("failed", "scene_setup_failed")
		_cleanup()
		quit(1)
		return

	ProjectSettings.set_setting("rendering/gaussian_splatting/debug/enable_pipeline_trace", true)
	ProjectSettings.set_setting("rendering/gaussian_splatting/debug/enable_data_logging", true)

	for i in range(90):
		await process_frame
		metrics["frames"] = i + 1
		if splat_node != null:
			renderer = splat_node.get_renderer()
		if renderer != null:
			break

	if renderer == null:
		var skip_reason := "Renderer unavailable (local RenderingDevice required)."
		_emit_metrics("skipped", skip_reason)
		print("%s %s" % [SKIP_MARKER, skip_reason])
		_cleanup()
		quit(0)
		return

	if renderer.has_method("set_debug_pipeline_trace_enabled"):
		renderer.set_debug_pipeline_trace_enabled(true)

	var recent_window_valid_samples := 0
	for frame in range(MAX_TEST_FRAMES):
		await process_frame
		metrics["frames"] = max(int(metrics.get("frames", 0)), 90 + frame + 1)
		var snapshot: Dictionary = renderer.get_pipeline_trace_snapshot()
		var data_flow: Dictionary = snapshot.get("data_flow", {})
		var recent_window: Dictionary = data_flow.get("recent_window", {})
		if not recent_window.is_empty():
			metrics["recent_window_seen"] = true
			recent_window_valid_samples += 1
			var capacity := int(recent_window.get("capacity", 0))
			var frames_recorded := int(recent_window.get("frames_recorded", 0))
			var frame_deltas: Array = recent_window.get("frame_deltas", [])
			metrics["recent_capacity"] = capacity
			metrics["recent_frames_recorded_max"] = max(int(metrics.get("recent_frames_recorded_max", 0)), frames_recorded)
			metrics["recent_frame_deltas_size_max"] = max(int(metrics.get("recent_frame_deltas_size_max", 0)), frame_deltas.size())
			metrics["recent_pack_sh_samples"] = int(recent_window.get("pack_sh_samples", 0))
			metrics["recent_pack_range_calls"] = int(recent_window.get("pack_range_calls", 0))

			var summed_pack_sh := _sum_delta_field(frame_deltas, "pack_sh_samples")
			var summed_pack_range := _sum_delta_field(frame_deltas, "pack_range_calls")
			if summed_pack_sh != int(recent_window.get("pack_sh_samples", -1)):
				_record_failure("recent_window pack_sh aggregate mismatch")
				_emit_metrics("failed", "recent_window_pack_sh_mismatch")
				_cleanup()
				quit(1)
				return
			if summed_pack_range != int(recent_window.get("pack_range_calls", -1)):
				_record_failure("recent_window pack_range aggregate mismatch")
				_emit_metrics("failed", "recent_window_pack_range_mismatch")
				_cleanup()
				quit(1)
				return

		metrics["lifetime_pack_sh_samples"] = int(data_flow.get("pack_sh_samples", 0))
		var pack_range_dict: Dictionary = data_flow.get("pack_range", {})
		metrics["lifetime_pack_range_calls"] = int(pack_range_dict.get("calls", 0))

		if recent_window_valid_samples >= REQUIRED_SAMPLES:
			break

	if not bool(metrics.get("recent_window_seen", false)):
		_record_failure("recent_window data was never emitted")
		_emit_metrics("failed", "recent_window_missing")
		_cleanup()
		quit(1)
		return

	var capacity_final := int(metrics.get("recent_capacity", 0))
	var deltas_max := int(metrics.get("recent_frame_deltas_size_max", 0))
	if capacity_final <= 0:
		_record_failure("recent_window capacity missing")
		_emit_metrics("failed", "recent_window_capacity_missing")
		_cleanup()
		quit(1)
		return
	if deltas_max > capacity_final:
		_record_failure("recent_window exceeded configured capacity")
		_emit_metrics("failed", "recent_window_capacity_exceeded")
		_cleanup()
		quit(1)
		return
	if int(metrics.get("lifetime_pack_sh_samples", 0)) <= 0:
		_record_failure("lifetime pack_sh_samples did not update")
		_emit_metrics("failed", "lifetime_pack_sh_missing")
		_cleanup()
		quit(1)
		return

	_emit_metrics("passed", "recent-window data-flow telemetry validated")
	_cleanup()
	quit(0)
