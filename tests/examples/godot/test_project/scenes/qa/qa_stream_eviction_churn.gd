extends "res://scripts/qa_test_base.gd"
## Streaming Eviction Churn Test: Forces chunk eviction under low budget.

@export var swap_interval_frames: int = 20
@export var near_position: Vector3 = Vector3(0.0, 5.0, 20.0)
@export var far_position: Vector3 = Vector3(0.0, 5.0, 80.0)

var camera: Camera3D
var world_node: GaussianSplatWorld3D
var use_near := false

var max_evicted := 0
var max_loaded := 0
var _monitors_missing := false
var _prev_settings := {}

func _ready():
	test_name = "Streaming Eviction Churn"
	test_duration = 8.0
	warmup_frames = 10
	super._ready()

	camera = get_node_or_null("Camera3D")
	world_node = get_node_or_null("World")

func _on_test_start():
	var keys = [
		"rendering/gaussian_splatting/streaming/route_policy",
		"rendering/gaussian_splatting/instance_pipeline/enabled",
		"rendering/gaussian_splatting/streaming/vram_budget_mb",
		"rendering/gaussian_splatting/streaming/auto_regulate_enabled",
		"rendering/gaussian_splatting/streaming/max_chunks_in_vram",
		"rendering/gaussian_splatting/streaming/min_chunks_in_vram",
	]
	for key in keys:
		if key == "rendering/gaussian_splatting/streaming/route_policy":
			_prev_settings[key] = int(ProjectSettings.get_setting(key, 1))
		elif key == "rendering/gaussian_splatting/instance_pipeline/enabled":
			_prev_settings[key] = ProjectSettings.get_setting(key, false)
		else:
			_prev_settings[key] = ProjectSettings.get_setting(key)

	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/route_policy", 1)
	ProjectSettings.set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", false)
	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/vram_budget_mb", 128)
	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/auto_regulate_enabled", true)
	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/max_chunks_in_vram", 8)
	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/min_chunks_in_vram", 2)
	if world_node != null:
		world_node.clear_world()
		world_node.apply_world()

	if camera != null:
		camera.global_position = far_position
		camera.look_at(Vector3.ZERO, Vector3.UP)

func _on_test_frame(_delta: float):
	if camera != null and frame_count % swap_interval_frames == 0:
		use_near = !use_near
		camera.global_position = near_position if use_near else far_position
		camera.look_at(Vector3.ZERO, Vector3.UP)

	var evicted = get_custom_monitor_value("gaussian_splatting/streaming_chunks_evicted_this_frame")
	var loaded = get_custom_monitor_value("gaussian_splatting/streaming_loaded_chunks")
	var vram_evicted = get_custom_monitor_value("gaussian_splatting/vram_evicted_this_frame")

	if evicted == null or loaded == null or vram_evicted == null:
		_monitors_missing = true
		return

	max_evicted = max(max_evicted, int(evicted), int(vram_evicted))
	max_loaded = max(max_loaded, int(loaded))

func _on_test_complete():
	result_metrics["max_evicted"] = max_evicted
	result_metrics["max_loaded"] = max_loaded
	for key in _prev_settings.keys():
		ProjectSettings.set_setting(key, _prev_settings[key])

	if _monitors_missing:
		_test_result = false
		_test_message = "Streaming monitors unavailable"
		return

	var ok = max_evicted > 0 and max_loaded > 0
	_test_result = ok
	_test_message = "evicted=%d loaded=%d" % [max_evicted, max_loaded]
