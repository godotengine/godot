extends "res://scripts/qa_test_base.gd"
## Streaming Multi-Asset Test: Ensures multiple world assets register and stream.

@export var capture_delay_frames: int = 12

const STREAMING_ROUTE_POLICY_PATH := "rendering/gaussian_splatting/streaming/route_policy"
const INSTANCE_PIPELINE_ENABLED_PATH := "rendering/gaussian_splatting/instance_pipeline/enabled"
const ROUTE_RESIDENT := 0
const ROUTE_STREAMING := 1

var world_a: GaussianSplatWorld3D
var world_b: GaussianSplatWorld3D
var manager: Object
var _captured := false
var _prev_streaming_settings := {}
var _capture_phase := 0
var _capture_error := ""

var _counts := {
	"chunk_a": 0,
	"chunk_b": 0,
	"count_a": 0,
	"count_b": 0,
	"total_gaussians": 0,
	"streaming_total_chunks": 0,
	"resident_route_uid": "",
	"resident_requested_route_policy": "",
	"resident_instance_backend_policy": "",
	"resident_backend_selection_reason": "",
	"resident_data_source": "",
	"resident_instance_contract_ready": false,
	"streaming_route_uid": "",
	"streaming_requested_route_policy": "",
	"streaming_instance_backend_policy": "",
	"streaming_backend_selection_reason": "",
	"streaming_data_source": "",
	"streaming_instance_contract_ready": false,
}

func _ready():
	test_name = "Streaming Multi-Asset"
	test_duration = 6.0
	warmup_frames = 10
	super._ready()

	world_a = get_node_or_null("WorldA")
	world_b = get_node_or_null("WorldB")
	if Engine.has_singleton("GaussianSplatManager"):
		manager = Engine.get_singleton("GaussianSplatManager")

func _on_test_start():
	_capture_error = ""
	_prev_streaming_settings[STREAMING_ROUTE_POLICY_PATH] = ProjectSettings.get_setting(
		STREAMING_ROUTE_POLICY_PATH, ROUTE_STREAMING
	)
	_prev_streaming_settings[INSTANCE_PIPELINE_ENABLED_PATH] = ProjectSettings.get_setting(
		INSTANCE_PIPELINE_ENABLED_PATH, false
	)
	ProjectSettings.set_setting(STREAMING_ROUTE_POLICY_PATH, ROUTE_RESIDENT)
	ProjectSettings.set_setting(INSTANCE_PIPELINE_ENABLED_PATH, true)
	if world_a != null:
		world_a.clear_world()
		world_a.apply_world()
	if world_b != null:
		world_b.clear_world()
	_capture_phase = 0

func _on_test_frame(_delta: float):
	if _captured:
		return
	if _capture_phase == 0 and frame_count < capture_delay_frames:
		return
	if _capture_phase == 1 and frame_count < capture_delay_frames * 2:
		return

	if _capture_phase == 0:
		if not _capture_world_snapshot(world_a, "resident"):
			_captured = true
			_finish_test()
			return
		ProjectSettings.set_setting(STREAMING_ROUTE_POLICY_PATH, ROUTE_STREAMING)
		if world_a != null:
			world_a.clear_world()
		if world_b != null:
			world_b.apply_world()
		_capture_phase = 1
		return

	if _capture_phase == 1:
		if not _capture_world_snapshot(world_b, "streaming"):
			_captured = true
			_finish_test()
			return

	if manager != null and manager.has_method("get_global_stats"):
		var stats: Dictionary = manager.get_global_stats()
		_counts["total_gaussians"] = int(stats.get("total_gaussians", 0))

	var total_chunks = get_custom_monitor_value("gaussian_splatting/streaming_total_chunks")
	if total_chunks != null:
		_counts["streaming_total_chunks"] = int(total_chunks)

	_captured = true
	_finish_test()

func _capture_world_snapshot(world: GaussianSplatWorld3D, prefix: String) -> bool:
	if world == null or world.world == null:
		_capture_error = "[RUNTIME_FAIL] Missing world resource for %s phase" % prefix
		return false

	var chunk_key := "chunk_a" if prefix == "resident" else "chunk_b"
	var count_key := "count_a" if prefix == "resident" else "count_b"
	_counts[chunk_key] = world.world.get_chunk_count()
	var data = world.world.get_gaussian_data()
	_counts[count_key] = data.get_count() if data != null else 0

	var renderer: Object = null
	if world.has_method("get_renderer"):
		renderer = world.get_renderer()
	if renderer == null or not renderer.has_method("get_render_stats"):
		_capture_error = "[RUNTIME_FAIL] Missing renderer stats for %s phase" % prefix
		return false

	var stats: Dictionary = renderer.get_render_stats()
	_counts["%s_route_uid" % prefix] = str(stats.get("route_uid", ""))
	_counts["%s_requested_route_policy" % prefix] = str(stats.get("requested_route_policy", ""))
	_counts["%s_instance_backend_policy" % prefix] = str(stats.get("instance_backend_policy", ""))
	_counts["%s_backend_selection_reason" % prefix] = str(stats.get("backend_selection_reason", ""))
	_counts["%s_data_source" % prefix] = str(stats.get("data_source", ""))
	_counts["%s_instance_contract_ready" % prefix] = bool(stats.get("instance_contract_ready", false))
	return true

func _route_matches(route_uid: String, expected_prefix: String) -> bool:
	return route_uid.begins_with("INSTANCE.%s" % expected_prefix.to_upper())

func _data_source_matches(data_source: String, expected_policy: String) -> bool:
	if expected_policy == "resident":
		return data_source == "ResidentInstanceAtlas" or data_source.findn("resident") != -1
	return data_source == "StreamingGPU" or data_source == "GPUBufferManager" or data_source.findn("stream") != -1

func _on_test_complete():
	for key in _counts.keys():
		result_metrics[key] = _counts[key]
	for key in _prev_streaming_settings.keys():
		ProjectSettings.set_setting(key, _prev_streaming_settings[key])

	if not _capture_error.is_empty():
		_test_result = false
		_test_message = _capture_error
		return

	var chunk_ok = _counts["chunk_a"] > 0 and _counts["chunk_b"] > 0
	var total_ok = _counts["total_gaussians"] >= max(_counts["count_a"], _counts["count_b"])
	var streaming_ok = _counts["streaming_total_chunks"] > 0
	var resident_ok = _route_matches(str(_counts["resident_route_uid"]), "resident") and _data_source_matches(str(_counts["resident_data_source"]), "resident")
	var streaming_route_ok = _route_matches(str(_counts["streaming_route_uid"]), "streaming")
	var streaming_source_ok = _data_source_matches(str(_counts["streaming_data_source"]), "streaming")
	var resident_policy_ok = _counts["resident_requested_route_policy"] == "resident"
	var streaming_policy_ok = _counts["streaming_requested_route_policy"] == "streaming"
	var resident_backend_ok = _counts["resident_instance_backend_policy"] == "resident"
	var streaming_backend_ok = _counts["streaming_instance_backend_policy"] == "streaming"
	var routing_ok = resident_ok and streaming_route_ok and streaming_source_ok and resident_policy_ok and streaming_policy_ok and resident_backend_ok and streaming_backend_ok
	var contract_ok = bool(_counts["resident_instance_contract_ready"]) and bool(_counts["streaming_instance_contract_ready"])

	_test_result = chunk_ok and total_ok and streaming_ok and routing_ok and contract_ok
	_test_message = "resident_route=%s streaming_route=%s chunks a=%d b=%d total_gaussians=%d" % [
		_counts["resident_route_uid"], _counts["streaming_route_uid"], _counts["chunk_a"], _counts["chunk_b"], _counts["total_gaussians"]
	]
