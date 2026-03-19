extends "res://scripts/qa_test_base.gd"
## Streaming Multi-Asset Test: Ensures multiple world assets register and stream.

@export var capture_delay_frames: int = 12

var world_a: GaussianSplatWorld3D
var world_b: GaussianSplatWorld3D
var manager: Object
var _captured := false
var _prev_streaming_settings := {}

var _counts := {
	"chunk_a": 0,
	"chunk_b": 0,
	"count_a": 0,
	"count_b": 0,
	"total_gaussians": 0,
	"streaming_total_chunks": 0,
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
	_prev_streaming_settings["rendering/gaussian_splatting/streaming/enabled"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/streaming/enabled", false
	)
	_prev_streaming_settings["rendering/gaussian_splatting/instance_pipeline/enabled"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/instance_pipeline/enabled", false
	)
	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/enabled", true)
	ProjectSettings.set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", false)
	if world_a != null:
		world_a.clear_world()
		world_a.apply_world()
	if world_b != null:
		world_b.clear_world()
		world_b.apply_world()

func _on_test_frame(_delta: float):
	if _captured:
		return
	if frame_count < capture_delay_frames:
		return

	if world_a != null and world_a.world != null:
		_counts["chunk_a"] = world_a.world.get_chunk_count()
		var data_a = world_a.world.get_gaussian_data()
		_counts["count_a"] = data_a.get_count() if data_a != null else 0
	if world_b != null and world_b.world != null:
		_counts["chunk_b"] = world_b.world.get_chunk_count()
		var data_b = world_b.world.get_gaussian_data()
		_counts["count_b"] = data_b.get_count() if data_b != null else 0

	if manager != null and manager.has_method("get_global_stats"):
		var stats: Dictionary = manager.get_global_stats()
		_counts["total_gaussians"] = int(stats.get("total_gaussians", 0))

	var total_chunks = get_custom_monitor_value("gaussian_splatting/streaming_total_chunks")
	if total_chunks != null:
		_counts["streaming_total_chunks"] = int(total_chunks)

	_captured = true
	_finish_test()

func _on_test_complete():
	for key in _counts.keys():
		result_metrics[key] = _counts[key]
	for key in _prev_streaming_settings.keys():
		ProjectSettings.set_setting(key, _prev_streaming_settings[key])

	var chunk_ok = _counts["chunk_a"] > 0 and _counts["chunk_b"] > 0
	var total_ok = _counts["total_gaussians"] >= max(_counts["count_a"], _counts["count_b"])
	var streaming_ok = _counts["streaming_total_chunks"] > 0

	_test_result = chunk_ok and total_ok and streaming_ok
	_test_message = "chunks a=%d b=%d total_gaussians=%d" % [
		_counts["chunk_a"], _counts["chunk_b"], _counts["total_gaussians"]
	]
