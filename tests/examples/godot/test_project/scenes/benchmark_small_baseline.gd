extends Node3D

const BenchmarkMetricsUtil = preload("res://scripts/benchmark_metrics.gd")
const BenchmarkSceneContract = preload("res://scripts/benchmark_scene_contract.gd")

const DEFAULT_BENCHMARK_DURATION := 20.0
const DEFAULT_OUTPUT_PATH := "user://benchmark_small_baseline_results.json"
const GRID_COLS := 1
const GRID_ROWS := 1
const GRID_SPACING := 1.0
const CAMERA_ORBIT_SPEED := 0.35

const MONITOR_KEYS := [
	"gpu_time_frame_ms",
	"gpu_time_cull_ms",
	"gpu_time_raster_ms",
	"visible_splats",
	"overflow_tile_count",
	"streaming_vram_usage_mb",
]

const PROJECT_SETTING_KEYS := [
	"rendering/gaussian_splatting/streaming/enabled",
	"rendering/gaussian_splatting/instance_pipeline/enabled",
	"rendering/gaussian_splatting/quality/tier_apply_streaming_budgets",
	"rendering/gaussian_splatting/animation/wind_enabled",
	"rendering/gaussian_splatting/animation/wind_strength",
	"rendering/gaussian_splatting/effects/max_effectors",
	"rendering/gaussian_splatting/effects/sphere_effector_enabled",
	"rendering/gaussian_splatting/lod/max_distance",
	"rendering/gaussian_splatting/lod/bias",
]

signal benchmark_scene_finished(result: Dictionary)

@onready var camera: Camera3D = $Camera3D
@onready var instance_root: Node3D = $InstanceRoot
@onready var performance_overlay: Control = $PerformanceOverlay

var benchmark_duration := DEFAULT_BENCHMARK_DURATION
var output_path := DEFAULT_OUTPUT_PATH
var headless_summary := false
var baseline_asset_path := ""

var _elapsed_s := 0.0
var _frame_ms: Array = []
var _fps: Array = []
var _monitor_max: Dictionary = {}
var _settings_snapshot: Dictionary = {}
var _settings_restored := false
var _result_report: Dictionary = {}
var _state_running := true
var _focus_point := Vector3.ZERO
var _primary_renderer_owner: Node3D = null
var _max_node_visible_splats := 0
var _pending_contract: Dictionary = {}
var _orchestrated := false

func apply_benchmark_contract(contract: Dictionary) -> void:
	_pending_contract = contract.duplicate(true)

func _ready() -> void:
	_apply_contract()
	_setup_runtime_state()
	_snapshot_project_settings()
	_apply_small_scene_settings()
	_build_instance_grid()
	_focus_point = _resolve_focus_point()
	performance_overlay.visible = true
	print("[BENCH-SMALL] Baseline benchmark start | duration=%.1fs | output=%s | headless_summary=%s" % [
		benchmark_duration,
		output_path,
		headless_summary,
	])

func _process(delta: float) -> void:
	if not _state_running:
		return

	if delta <= 0.0:
		return

	_elapsed_s += delta
	_update_camera()
	_sample_metrics(delta)

	if _elapsed_s >= benchmark_duration:
		_finish_benchmark()

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_ESCAPE:
			_restore_project_settings()
			get_tree().quit(0)
		elif event.keycode == KEY_R and not _state_running:
			get_tree().reload_current_scene()

func _exit_tree() -> void:
	_restore_project_settings()

func _apply_contract() -> void:
	var scene_id := BenchmarkSceneContract.scene_id_from_path(get_tree().current_scene.scene_file_path)
	var defaults := {
		"duration_s": DEFAULT_BENCHMARK_DURATION,
		"output_path": DEFAULT_OUTPUT_PATH,
		"headless_summary": false,
		"asset_path": "",
		"lane_id": "small_baseline",
	}
	var contract := BenchmarkSceneContract.resolve_contract(scene_id, defaults, _pending_contract)
	_orchestrated = bool(contract.get("orchestrated", false))
	benchmark_duration = max(5.0, float(contract.get("duration_s", DEFAULT_BENCHMARK_DURATION)))
	output_path = str(contract.get("output_path", DEFAULT_OUTPUT_PATH))
	headless_summary = bool(contract.get("headless_summary", false))
	baseline_asset_path = BenchmarkSceneContract.resolve_asset_path(
		scene_id,
		str(contract.get("lane_id", "small_baseline")),
		str(contract.get("asset_path", "")),
		"",
	)

func _setup_runtime_state() -> void:
	OS.set_low_processor_usage_mode(false)
	OS.set_low_processor_usage_mode_sleep_usec(0)
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

func _build_instance_grid() -> void:
	for child in instance_root.get_children():
		child.queue_free()

	_primary_renderer_owner = null
	for row in range(GRID_ROWS):
		for col in range(GRID_COLS):
			var node := GaussianSplatNode3D.new()
			node.name = "Baseline_%02d_%02d" % [row, col]
			node.ply_file_path = baseline_asset_path
			node.position = Vector3(
				(float(col) - float(GRID_COLS - 1) * 0.5) * GRID_SPACING,
				0.0,
				(float(row) - float(GRID_ROWS - 1) * 0.5) * GRID_SPACING
			)
			node.scale = Vector3.ONE
			node.set("quality/preset", 2)
			node.set("quality/max_splat_count", 50000)
			node.set("quality/lod_bias", 1.2)
			node.set("quality/max_render_distance", 400.0)
			instance_root.add_child(node)
			if _primary_renderer_owner == null:
				_primary_renderer_owner = node

func _resolve_focus_point() -> Vector3:
	return instance_root.global_position + Vector3(0.0, 1.0, 0.0)

func _update_camera() -> void:
	var orbit_radius := maxf(float(max(GRID_COLS, GRID_ROWS)) * GRID_SPACING * 0.9, 30.0)
	var angle := _elapsed_s * CAMERA_ORBIT_SPEED
	camera.global_position = _focus_point + Vector3(cos(angle) * orbit_radius, 12.0, sin(angle) * orbit_radius)
	camera.look_at(_focus_point, Vector3.UP)

func _sample_metrics(delta: float) -> void:
	var frame_ms := delta * 1000.0
	var fps := 1.0 / delta
	_frame_ms.append(frame_ms)
	_fps.append(fps)

	for key in MONITOR_KEYS:
		var monitor_id := "gaussian_splatting/%s" % key
		if not Performance.has_custom_monitor(monitor_id):
			continue
		var value := float(Performance.get_custom_monitor(monitor_id))
		if not _monitor_max.has(key) or value > float(_monitor_max[key]):
			_monitor_max[key] = value

	if _primary_renderer_owner != null and _primary_renderer_owner.has_method("get_visible_splat_count"):
		var visible := int(_primary_renderer_owner.get_visible_splat_count())
		if visible > _max_node_visible_splats:
			_max_node_visible_splats = visible

func _get_primary_renderer():
	if _primary_renderer_owner != null and _primary_renderer_owner.has_method("get_renderer"):
		return _primary_renderer_owner.get_renderer()
	return null

func _finish_benchmark() -> void:
	_state_running = false
	_result_report = _build_report()
	_restore_project_settings()
	_write_report(_result_report)

	if _orchestrated:
		emit_signal("benchmark_scene_finished", {
			"success": true,
			"lane_id": "small_baseline",
			"report_path": output_path,
			"report": _result_report,
		})
		queue_free()
		return

	if headless_summary or DisplayServer.get_name() == "headless":
		_print_headless_summary(_result_report)
		get_tree().quit(0)
		return

	print("[BENCH-SMALL] Benchmark complete. Press Esc to quit, R to rerun.")

func _build_report() -> Dictionary:
	var overall: Dictionary = BenchmarkMetricsUtil.summarize_samples(_frame_ms, _fps)
	var renderer_stats: Dictionary = {}
	var renderer = _get_primary_renderer()
	if renderer != null and renderer.has_method("get_render_stats"):
		var stats = renderer.get_render_stats()
		if stats is Dictionary:
			renderer_stats = stats

	overall["route_uid"] = renderer_stats.get("route_uid", "")
	overall["sort_route_uid"] = renderer_stats.get("sort_route_uid", "")
	overall["data_source"] = renderer_stats.get("data_source", "")
	overall["stage_cull_status"] = renderer_stats.get("stage_cull_status", "")
	overall["stage_sort_status"] = renderer_stats.get("stage_sort_status", "")
	overall["stage_raster_status"] = renderer_stats.get("stage_raster_status", "")

	var settings_now := _collect_current_project_settings()
	var score: float = BenchmarkMetricsUtil.compute_score(overall, _monitor_max)
	var recommendations: Array[Dictionary] = BenchmarkMetricsUtil.build_recommendations({
		"overall": overall,
		"monitor_max": _monitor_max,
		"project_settings": settings_now,
	})

	return {
		"name": "GodotGS Small Baseline Benchmark",
		"scene": "res://scenes/benchmark_small_baseline.tscn",
		"duration_s": _elapsed_s,
		"configured_duration_s": benchmark_duration,
		"output_path": output_path,
		"timestamp_unix": Time.get_unix_time_from_system(),
		"platform": OS.get_name(),
		"instance_count": GRID_COLS * GRID_ROWS,
		"baseline_ply_path": baseline_asset_path,
		"node_visible_splats_max": _max_node_visible_splats,
		"overall": overall,
		"monitor_max": _monitor_max,
		"project_settings": settings_now,
		"score": score,
		"recommendations": recommendations,
	}

func _write_report(report: Dictionary) -> void:
	var file := FileAccess.open(output_path, FileAccess.WRITE)
	if file == null:
		push_warning("[BENCH-SMALL] Failed to write report: %s" % output_path)
		return
	file.store_string(JSON.stringify(report, "  "))
	print("[BENCH-SMALL] Report written to %s" % output_path)

func _print_headless_summary(report: Dictionary) -> void:
	var overall: Dictionary = report.get("overall", {})
	print("[BENCH-SMALL] Summary | score=%.1f avg_fps=%.1f p1_fps=%.1f p99_ms=%.2f" % [
		float(report.get("score", 0.0)),
		float(overall.get("avg_fps", 0.0)),
		float(overall.get("p1_fps", 0.0)),
		float(overall.get("p99_frame_ms", 0.0)),
	])
	print("[BENCH-SMALL] Visibility | node_visible_splats_max=%d monitor_visible_splats_max=%.1f" % [
		int(report.get("node_visible_splats_max", 0)),
		float(report.get("monitor_max", {}).get("visible_splats", 0.0)),
	])

func _snapshot_project_settings() -> void:
	for key in PROJECT_SETTING_KEYS:
		var existed := ProjectSettings.has_setting(key)
		var value = null
		if existed:
			value = ProjectSettings.get_setting(key)
		_settings_snapshot[key] = {
			"existed": existed,
			"value": value,
		}

func _apply_small_scene_settings() -> void:
	_set_project_setting("rendering/gaussian_splatting/streaming/enabled", false)
	_set_project_setting("rendering/gaussian_splatting/instance_pipeline/enabled", false)
	_set_project_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", false)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_strength", 0.0)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/lod/max_distance", 500.0)
	_set_project_setting("rendering/gaussian_splatting/lod/bias", 1.0)

func _restore_project_settings() -> void:
	if _settings_restored:
		return
	_settings_restored = true
	for key in _settings_snapshot.keys():
		var snapshot: Dictionary = _settings_snapshot[key]
		if bool(snapshot.get("existed", false)):
			ProjectSettings.set_setting(key, snapshot.get("value"))

func _collect_current_project_settings() -> Dictionary:
	var out := {}
	for key in PROJECT_SETTING_KEYS:
		if ProjectSettings.has_setting(key):
			out[key] = ProjectSettings.get_setting(key)
	return out

func _set_project_setting(key: String, value) -> void:
	ProjectSettings.set_setting(key, value)
