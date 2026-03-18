extends "res://scripts/qa_test_base.gd"
## Static Fast-Path Test: Verifies identity flags recorded for static instances.

@export var capture_delay_frames: int = 10

var renderer: Object
var capture_frame_count := 0
var _result_ready := false
var _result_message := ""
var _result_pass := false
var _prev_debug_settings := {}

func _ready():
	test_name = "Static Fast-Path"
	test_duration = 5.0
	warmup_frames = 10
	super._ready()

func _on_test_start():
	_prev_debug_settings["rendering/gaussian_splatting/debug/enable_pipeline_trace"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/debug/enable_pipeline_trace"
	)
	_prev_debug_settings["rendering/gaussian_splatting/debug/enable_data_logging"] = ProjectSettings.get_setting(
		"rendering/gaussian_splatting/debug/enable_data_logging"
	)

	ProjectSettings.set_setting("rendering/gaussian_splatting/debug/enable_pipeline_trace", true)
	ProjectSettings.set_setting("rendering/gaussian_splatting/debug/enable_data_logging", true)
	renderer = get_gs_renderer("StaticInstance")
	if renderer != null and renderer.has_method("set_debug_pipeline_trace_enabled"):
		renderer.set_debug_pipeline_trace_enabled(true)
	if renderer == null or not renderer.has_method("get_pipeline_trace_snapshot"):
		_result_pass = false
		_result_message = "Renderer missing or pipeline trace unavailable"
		_result_ready = true
		_finish_test()
		return
	capture_frame_count = 0

func _on_test_frame(_delta: float):
	if renderer == null:
		return
	capture_frame_count += 1
	if capture_frame_count < capture_delay_frames:
		return

	var snapshot: Dictionary = renderer.get_pipeline_trace_snapshot()
	var data_flow: Dictionary = snapshot.get("data_flow", {})
	var flags: Dictionary = data_flow.get("instance_flags", {})
	if flags.is_empty():
		_result_pass = false
		_result_message = "No instance flag data in pipeline trace"
		_result_ready = true
		_finish_test()
		return

	var total = int(flags.get("total", 0))
	var rot_identity = int(flags.get("rotation_identity", 0))
	var scale_identity = int(flags.get("scale_identity", 0))
	var translation_zero = int(flags.get("translation_zero", 0))
	var fully_identity = int(flags.get("fully_identity", 0))

	result_metrics["total_instances"] = total
	result_metrics["rotation_identity"] = rot_identity
	result_metrics["scale_identity"] = scale_identity
	result_metrics["translation_zero"] = translation_zero
	result_metrics["fully_identity"] = fully_identity

	var has_static = fully_identity >= 1
	var has_translation_zero = translation_zero >= 1
	var counts_ok = total >= 1 and rot_identity >= 1 and scale_identity >= 1

	_result_pass = has_static and has_translation_zero and counts_ok
	_result_message = "flags total=%d rot=%d scale=%d trans0=%d full=%d" % [total, rot_identity, scale_identity, translation_zero, fully_identity]
	_result_ready = true
	_finish_test()

func _on_test_complete():
	for key in _prev_debug_settings.keys():
		ProjectSettings.set_setting(key, _prev_debug_settings[key])
	if _result_ready:
		_test_result = _result_pass
		_test_message = _result_message
	else:
		_test_result = false
		_test_message = "No result captured"
