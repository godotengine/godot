extends Node3D

# 50-instance spinning stress test for instance pipeline
const TEST_DURATION := 180.0
const INSTANCE_COUNT := 50
const GRID_COLS := 10
const GRID_SPACING := 25.0

var _instances: Array[Node3D] = []
var _frame_count := 0
var _last_fps_time := 0.0
var _fps_samples := []
var _total_time := 0.0

func _ready() -> void:
	OS.set_low_processor_usage_mode(false)
	OS.set_low_processor_usage_mode_sleep_usec(0)
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

	var asset = load("res://tests/fixtures/test_splats.ply")
	if not asset:
		push_error("[PERF-50] Failed to load synthetic fixture asset: res://tests/fixtures/test_splats.ply")
		get_tree().quit()
		return

	for i in INSTANCE_COUNT:
		var col := i % GRID_COLS
		var row := i / GRID_COLS
		var node := GaussianSplatNode3D.new()
		node.name = "Instance_%02d" % i
		node.transform.origin = Vector3(col * GRID_SPACING, 0, row * GRID_SPACING)
		add_child(node)
		node.splat_asset = asset
		_instances.append(node)

	# Point camera at center of grid
	var cam := get_node_or_null("Camera3D")
	if cam:
		var center_x := (GRID_COLS - 1) * GRID_SPACING / 2.0
		var rows := (INSTANCE_COUNT - 1) / GRID_COLS
		var center_z := rows * GRID_SPACING / 2.0
		cam.look_at(Vector3(center_x, 0, center_z), Vector3.UP)

	var instance_enabled = ProjectSettings.get_setting("rendering/gaussian_splatting/instance_pipeline/enabled", false)
	print("[PERF-50] Spawned %d instances (%dx%d grid, %.0f spacing)" % [
		INSTANCE_COUNT, GRID_COLS, ceili(float(INSTANCE_COUNT) / GRID_COLS), GRID_SPACING])
	print("[PERF-50] Instance pipeline: %s | Duration: %ds" % [instance_enabled, int(TEST_DURATION)])

func _process(delta: float) -> void:
	_frame_count += 1
	_last_fps_time += delta
	_total_time += delta

	# Spin all instances
	for inst in _instances:
		inst.rotate_y(delta * 0.5)

	if _last_fps_time >= 1.0:
		var fps := _frame_count / _last_fps_time
		_fps_samples.append(fps)
		var avg := _get_avg_fps()
		var min_fps: float = _fps_samples.min()
		var max_fps: float = _fps_samples.max()
		print("[PERF-50] FPS: %.1f | avg: %.1f | min: %.1f | max: %.1f | t=%ds" % [
			fps, avg, min_fps, max_fps, int(_total_time)])
		_frame_count = 0
		_last_fps_time = 0.0

	if _total_time >= TEST_DURATION:
		print("[PERF-50] === FINAL: avg=%.1f min=%.1f max=%.1f over %d samples ===" % [
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
		print("[PERF-50] === EARLY EXIT: avg=%.1f over %d samples ===" % [_get_avg_fps(), _fps_samples.size()])
		get_tree().quit()
