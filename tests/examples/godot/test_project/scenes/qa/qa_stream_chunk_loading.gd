extends "res://scripts/qa_test_base.gd"
## Streaming Chunk Loading Test: Verifies chunks load when camera moves into range.

@export var move_delay_frames: int = 10
@export var far_position: Vector3 = Vector3(0.0, 5.0, 80.0)
@export var near_position: Vector3 = Vector3(0.0, 5.0, 25.0)

var world_node: GaussianSplatWorld3D
var camera: Camera3D

var max_loaded_chunks := 0
var max_loaded_this_frame := 0
var max_visible_chunks := 0
var max_total_chunks := 0
var _monitor_ready_seen := false
var _monitors_missing_after_ready := false
var _prev_streaming_settings := {}

func _ready():
	test_name = "Streaming Chunk Loading"
	test_duration = 6.0
	warmup_frames = 10
	super._ready()

	world_node = get_node_or_null("World")
	camera = get_node_or_null("Camera3D")

func _on_test_start():
	_prev_streaming_settings["rendering/gaussian_splatting/streaming/enabled"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/streaming/enabled", false
	)
	_prev_streaming_settings["rendering/gaussian_splatting/instance_pipeline/enabled"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/instance_pipeline/enabled", false
	)
	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/enabled", true)
	ProjectSettings.set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", false)
	if world_node != null:
		world_node.clear_world()
		world_node.apply_world()
	if camera != null:
		camera.global_position = far_position
		camera.look_at(Vector3.ZERO, Vector3.UP)

func _on_test_frame(_delta: float):
	if frame_count == move_delay_frames and camera != null:
		camera.global_position = near_position
		camera.look_at(Vector3.ZERO, Vector3.UP)

	var total_chunks = get_custom_monitor_value("gaussian_splatting/streaming_total_chunks")
	var loaded_chunks = get_custom_monitor_value("gaussian_splatting/streaming_loaded_chunks")
	var visible_chunks = get_custom_monitor_value("gaussian_splatting/streaming_visible_chunks")
	var loaded_this_frame = get_custom_monitor_value("gaussian_splatting/streaming_chunks_loaded_this_frame")
	var streaming_monitor_ready = get_custom_monitor_value("gaussian_splatting/streaming_monitor_ready")
	var monitor_ready = streaming_monitor_ready != null and int(streaming_monitor_ready) > 0

	if not monitor_ready:
		return
	_monitor_ready_seen = true
	if total_chunks == null or loaded_chunks == null or visible_chunks == null or loaded_this_frame == null:
		_monitors_missing_after_ready = true
		return

	max_total_chunks = max(max_total_chunks, int(total_chunks))
	max_loaded_chunks = max(max_loaded_chunks, int(loaded_chunks))
	max_visible_chunks = max(max_visible_chunks, int(visible_chunks))
	max_loaded_this_frame = max(max_loaded_this_frame, int(loaded_this_frame))

func _on_test_complete():
	result_metrics["total_chunks_max"] = max_total_chunks
	result_metrics["loaded_chunks_max"] = max_loaded_chunks
	result_metrics["visible_chunks_max"] = max_visible_chunks
	result_metrics["loaded_this_frame_max"] = max_loaded_this_frame
	for key in _prev_streaming_settings.keys():
		ProjectSettings.set_setting(key, _prev_streaming_settings[key])

	if world_node != null and world_node.world != null:
		result_metrics["asset_chunk_count"] = world_node.world.get_chunk_count()

	if not _monitor_ready_seen:
		_test_result = false
		_test_message = "Streaming monitor never became ready"
		return

	if _monitors_missing_after_ready:
		_test_result = false
		_test_message = "Streaming monitors unavailable after readiness"
		return

	var ok = max_total_chunks > 0 and max_loaded_chunks > 0 and max_loaded_this_frame > 0
	_test_result = ok
	_test_message = "chunks total=%d loaded=%d loaded_this_frame=%d" % [max_total_chunks, max_loaded_chunks, max_loaded_this_frame]
