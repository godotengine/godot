extends Node3D

# Clean performance baseline - always prints FPS, auto-quits after duration
const TEST_DURATION := 30.0
const TRACE_DUMP_ENABLED := true
const TRACE_DUMP_TIME := 3.0
const TRACE_WARMUP_TIME := 0.5

var _frame_count := 0
var _last_fps_time := 0.0
var _fps_samples := []
var _total_time := 0.0

var _trace_dumped := false
var _trace_enabled := false

func _ready() -> void:
	# Force full-speed rendering in editor/test runs.
	OS.set_low_processor_usage_mode(false)
	OS.set_low_processor_usage_mode_sleep_usec(0)
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	var instance_enabled = ProjectSettings.get_setting("rendering/gaussian_splatting/instance_pipeline/enabled", false)
	var compute_raster = ProjectSettings.get_setting("rendering/gaussian_splatting/gpu_sorting/enable_compute_raster", false)
	print("[PERF] Instance pipeline: %s | Compute raster: %s | Resolution: %dx%d | Duration: %ds" % [
		instance_enabled, compute_raster, get_viewport().size.x, get_viewport().size.y, int(TEST_DURATION)])
	print("[PERF] max_fps=%d low_proc=%s vsync=%d" % [
		Engine.max_fps,
		"true" if OS.is_in_low_processor_usage_mode() else "false",
		DisplayServer.window_get_vsync_mode()
	])
	# Point camera at midpoint between both cabins
	var cam = get_node_or_null("Camera3D")
	if cam:
		cam.look_at(Vector3(17.1, 0, 0), Vector3.UP)
	# Trace is enabled on-demand to avoid per-frame GPU readback stalls.

func _process(delta: float) -> void:
	_frame_count += 1
	_last_fps_time += delta
	_total_time += delta

	# Rotate SecondaryAsset to verify instance pipeline handles dynamic transforms
	var secondary = get_node_or_null("SecondaryAsset")
	if secondary:
		secondary.rotate_y(delta * 0.5)

	# Dump trace after warmup (streaming loaded, rendering stable)
	if TRACE_DUMP_ENABLED:
		var primary = get_node_or_null("PrimaryAsset")
		if primary:
			var r = primary.get_renderer()
			if r:
				if not _trace_enabled and _total_time >= max(0.0, TRACE_DUMP_TIME - TRACE_WARMUP_TIME):
					_trace_enabled = true
					r.set_debug_pipeline_trace_enabled(true)
				if _total_time >= TRACE_DUMP_TIME and not _trace_dumped:
					_trace_dumped = true
					r.dump_pipeline_trace_to_file("user://gs_raster_path_check.json")
					r.set_debug_pipeline_trace_enabled(false)
					_trace_enabled = false
					print("[PERF] Pipeline trace dumped + trace DISABLED (no more readback stalls)")

	if _last_fps_time >= 1.0:
		var fps = _frame_count / _last_fps_time
		_fps_samples.append(fps)
		var avg = _get_avg_fps()
		var min_fps = _fps_samples.min()
		var max_fps = _fps_samples.max()
		var engine_fps = Engine.get_frames_per_second()
		var process_frames = Engine.get_process_frames()
		var render_frames = Engine.get_frames_drawn()
		print("[PERF] FPS: %.1f | engine_fps=%.1f | process_frames=%d | render_frames=%d | avg: %.1f | min: %.1f | max: %.1f | t=%ds" % [
			fps, engine_fps, process_frames, render_frames, avg, min_fps, max_fps, int(_total_time)])
		_frame_count = 0
		_last_fps_time = 0.0

	if _total_time >= TEST_DURATION:
		print("[PERF] === FINAL: avg=%.1f min=%.1f max=%.1f over %d samples ===" % [
			_get_avg_fps(), _fps_samples.min() if not _fps_samples.is_empty() else 0.0,
			_fps_samples.max() if not _fps_samples.is_empty() else 0.0, _fps_samples.size()])
		get_tree().quit()

func _get_avg_fps() -> float:
	if _fps_samples.is_empty():
		return 0.0
	var total := 0.0
	for sample in _fps_samples:
		total += sample
	return total / _fps_samples.size()

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		print("[PERF] === EARLY EXIT: avg=%.1f over %d samples ===" % [_get_avg_fps(), _fps_samples.size()])
		get_tree().quit()
