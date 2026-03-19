extends Node3D

const BenchmarkMetricsUtil = preload("res://scripts/benchmark_metrics.gd")
const BenchmarkSceneContract = preload("res://scripts/benchmark_scene_contract.gd")

const DEFAULT_BENCHMARK_DURATION := 180.0
const DEFAULT_OUTPUT_PATH := "user://benchmark_unified_results.json"
const SETTINGS_APPLY_INTERVAL := 0.1
const INSTANCE_GRID_COLS := 6
const INSTANCE_GRID_ROWS := 4
const INSTANCE_SPACING := 7.0

const PHASE_TEMPLATE := [
	{"name": "Warmup", "start": 0.0, "end": 0.083333, "setup": "_setup_phase_warmup"},
	{"name": "Instance", "start": 0.083333, "end": 0.25, "setup": "_setup_phase_instance"},
	{"name": "Streaming", "start": 0.25, "end": 0.5, "setup": "_setup_phase_streaming"},
	{"name": "Lighting", "start": 0.5, "end": 0.666666, "setup": "_setup_phase_lighting"},
	{"name": "Effects", "start": 0.666666, "end": 0.833333, "setup": "_setup_phase_effects"},
	{"name": "LOD", "start": 0.833333, "end": 0.972222, "setup": "_setup_phase_lod"},
	{"name": "Finalize", "start": 0.972222, "end": 1.0, "setup": "_setup_phase_finalize"},
]

const CAMERA_KEYS := [
	{"t": 0.0, "offset": Vector3(0.0, 16.0, 45.0), "look": Vector3(0.0, 6.0, 0.0)},
	{"t": 0.18, "offset": Vector3(26.0, 14.0, 32.0), "look": Vector3(0.0, 5.0, 0.0)},
	{"t": 0.36, "offset": Vector3(40.0, 20.0, 4.0), "look": Vector3(0.0, 6.0, 0.0)},
	{"t": 0.52, "offset": Vector3(22.0, 13.0, -26.0), "look": Vector3(0.0, 5.0, 0.0)},
	{"t": 0.68, "offset": Vector3(0.0, 11.0, -15.0), "look": Vector3(0.0, 4.0, 0.0)},
	{"t": 0.84, "offset": Vector3(-24.0, 17.0, -20.0), "look": Vector3(0.0, 6.0, 0.0)},
	{"t": 1.0, "offset": Vector3(0.0, 16.0, 45.0), "look": Vector3(0.0, 6.0, 0.0)},
]

const MONITOR_KEYS := [
	"gpu_time_frame_ms",
	"gpu_time_cull_ms",
	"gpu_time_raster_ms",
	"visible_splats",
	"overflow_tile_count",
	"streaming_total_chunks",
	"streaming_visible_chunks",
	"streaming_loaded_chunks",
	"streaming_vram_usage_mb",
	"streaming_chunks_loaded_this_frame",
	"streaming_chunks_evicted_this_frame",
	"streaming_visible_count",
	"streaming_visible_change_ratio",
	"streaming_upload_bandwidth_cap_hit",
	"streaming_chunk_load_cap_hit",
	"streaming_queue_pressure_active",
	"lod_current_level",
	"lod_transitions_this_frame",
	"lod_reduction_ratio_pct",
]

const PROJECT_SETTING_KEYS := [
	"rendering/gaussian_splatting/streaming/enabled",
	"rendering/gaussian_splatting/instance_pipeline/enabled",
	"rendering/gaussian_splatting/quality/tier_apply_streaming_budgets",
	"rendering/gaussian_splatting/streaming/vram_budget_mb",
	"rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame",
	"rendering/gaussian_splatting/streaming/pack_worker_threads",
	"rendering/gaussian_splatting/lod/max_distance",
	"rendering/gaussian_splatting/lod/bias",
	"rendering/gaussian_splatting/lod/hysteresis_zone",
	"rendering/gaussian_splatting/lighting/direct_light_scale",
	"rendering/gaussian_splatting/lighting/indirect_sh_scale",
	"rendering/gaussian_splatting/lighting/shadow_strength",
	"rendering/gaussian_splatting/animation/wind_enabled",
	"rendering/gaussian_splatting/animation/wind_strength",
	"rendering/gaussian_splatting/animation/wind_frequency",
	"rendering/gaussian_splatting/animation/wind_direction_x",
	"rendering/gaussian_splatting/animation/wind_direction_y",
	"rendering/gaussian_splatting/animation/wind_direction_z",
	"rendering/gaussian_splatting/effects/max_effectors",
	"rendering/gaussian_splatting/effects/sphere_effector_enabled",
	"rendering/gaussian_splatting/effects/sphere_effector_center_x",
	"rendering/gaussian_splatting/effects/sphere_effector_center_y",
	"rendering/gaussian_splatting/effects/sphere_effector_center_z",
	"rendering/gaussian_splatting/effects/sphere_effector_radius",
	"rendering/gaussian_splatting/effects/sphere_effector_strength",
	"rendering/gaussian_splatting/effects/sphere_effector_falloff",
	"rendering/gaussian_splatting/effects/sphere_effector_frequency",
]

signal benchmark_scene_finished(result: Dictionary)

@onready var camera: Camera3D = $Camera3D
@onready var performance_overlay: Control = $PerformanceOverlay
@onready var results_panel: CanvasLayer = $BenchmarkResultsPanel
@onready var world_environment: WorldEnvironment = $WorldEnvironment
@onready var streaming_world: Node3D = $StreamingWorld
@onready var instance_root: Node3D = $InstanceRoot
@onready var hero_effect_asset: Node3D = $HeroEffectsAsset
@onready var directional_light: DirectionalLight3D = $DirectionalLight3D
@onready var omni_light_a: OmniLight3D = $OmniLightA
@onready var omni_light_b: OmniLight3D = $OmniLightB
@onready var spot_light: SpotLight3D = $SpotLight

var benchmark_duration := DEFAULT_BENCHMARK_DURATION
var output_path := DEFAULT_OUTPUT_PATH
var headless_summary := false
var instance_asset_path := ""

var _elapsed_s := 0.0
var _phase_index := -1
var _phase_name := ""
var _phase_definitions: Array = []
var _phase_runtime: Dictionary = {}
var _overall_fps: Array = []
var _overall_frame_ms: Array = []
var _monitor_max: Dictionary = {}
var _last_renderer_stats: Dictionary = {}
var _instance_nodes: Array = []
var _instance_asset: Resource = null
var _focus_point := Vector3.ZERO
var _settings_snapshot: Dictionary = {}
var _settings_restored := false
var _result_report: Dictionary = {}
var _settings_apply_accum := 0.0

var _state_running := true
var _pending_contract: Dictionary = {}
var _orchestrated := false

func apply_benchmark_contract(contract: Dictionary) -> void:
	_pending_contract = contract.duplicate(true)

func _ready() -> void:
	_apply_contract()
	_setup_runtime_state()
	_snapshot_project_settings()
	_apply_base_project_settings()
	_build_phase_definitions()
	_build_instance_grid()
	_focus_point = _resolve_focus_point()
	results_panel.visible = false
	performance_overlay.visible = true
	print("[BENCH] Unified benchmark start | duration=%.1fs | output=%s | headless_summary=%s" % [
		benchmark_duration,
		output_path,
		headless_summary,
	])

func _process(delta: float) -> void:
	if not _state_running:
		return

	_elapsed_s += delta
	_settings_apply_accum += delta

	_update_phase()
	_update_camera()
	_animate_scene(delta)
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
		"lane_id": "unified_composite",
	}
	var contract := BenchmarkSceneContract.resolve_contract(scene_id, defaults, _pending_contract)
	_orchestrated = bool(contract.get("orchestrated", false))
	benchmark_duration = max(5.0, float(contract.get("duration_s", DEFAULT_BENCHMARK_DURATION)))
	output_path = str(contract.get("output_path", DEFAULT_OUTPUT_PATH))
	headless_summary = bool(contract.get("headless_summary", false))
	instance_asset_path = BenchmarkSceneContract.resolve_asset_path(
		scene_id,
		str(contract.get("lane_id", "unified_composite")),
		str(contract.get("asset_path", "")),
		"",
	)

func _setup_runtime_state() -> void:
	OS.set_low_processor_usage_mode(false)
	OS.set_low_processor_usage_mode_sleep_usec(0)
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	_instance_asset = load(instance_asset_path)
	if _instance_asset == null:
		push_warning("[BENCH] Missing instance asset: %s" % instance_asset_path)
	elif hero_effect_asset != null:
		hero_effect_asset.set("splat_asset", _instance_asset)

func _build_phase_definitions() -> void:
	_phase_definitions.clear()
	_phase_runtime.clear()
	for template_variant in PHASE_TEMPLATE:
		var template: Dictionary = template_variant
		var name := str(template.get("name", "Phase"))
		var start_s := float(template.get("start", 0.0)) * benchmark_duration
		var end_s := float(template.get("end", 1.0)) * benchmark_duration
		var setup := str(template.get("setup", ""))
		_phase_definitions.append({
			"name": name,
			"start_s": start_s,
			"end_s": end_s,
			"setup": setup,
		})
		_phase_runtime[name] = {
			"frame_ms": [],
			"fps": [],
			"monitor_max": {},
			"renderer_stats": {},
		}

func _build_instance_grid() -> void:
	for child in instance_root.get_children():
		child.queue_free()
	_instance_nodes.clear()
	if _instance_asset == null:
		return

	for row in range(INSTANCE_GRID_ROWS):
		for col in range(INSTANCE_GRID_COLS):
			var node := GaussianSplatNode3D.new()
			node.name = "Instance_%02d_%02d" % [row, col]
			node.splat_asset = _instance_asset
			node.position = Vector3((col - float(INSTANCE_GRID_COLS - 1) * 0.5) * INSTANCE_SPACING, 0.0, -18.0 + row * INSTANCE_SPACING)
			node.set("quality/preset", 2)
			node.set("quality/lod_bias", 0.8)
			instance_root.add_child(node)
			_instance_nodes.append(node)

func _resolve_focus_point() -> Vector3:
	if streaming_world != null and streaming_world.has_method("get_world"):
		var world_res = streaming_world.get_world()
		if world_res != null and world_res.has_method("get_bounds"):
			var bounds = world_res.get_bounds()
			if bounds is AABB and bounds.has_volume():
				return bounds.get_center()
	return Vector3.ZERO

func _update_phase() -> void:
	var new_index := _phase_definitions.size() - 1
	for i in range(_phase_definitions.size()):
		var phase: Dictionary = _phase_definitions[i]
		if _elapsed_s < float(phase.get("end_s", benchmark_duration)):
			new_index = i
			break

	if new_index == _phase_index:
		return

	_phase_index = new_index
	var active_phase: Dictionary = _phase_definitions[_phase_index]
	_phase_name = str(active_phase.get("name", "Phase"))
	var setup_method := str(active_phase.get("setup", ""))
	if not setup_method.is_empty() and has_method(setup_method):
		call(setup_method)
	print("[BENCH] Enter phase: %s (t=%.1fs)" % [_phase_name, _elapsed_s])

func _update_camera() -> void:
	var t := clampf(_elapsed_s / benchmark_duration, 0.0, 1.0)
	var sample := _sample_camera_track(t)
	var camera_pos: Vector3 = _focus_point + sample.get("offset", Vector3(0, 12, 30))
	var look_pos: Vector3 = _focus_point + sample.get("look", Vector3(0, 5, 0))
	camera.global_position = camera_pos
	camera.look_at(look_pos, Vector3.UP)

func _sample_camera_track(t: float) -> Dictionary:
	if CAMERA_KEYS.size() == 1:
		return CAMERA_KEYS[0]
	if t <= float(CAMERA_KEYS[0]["t"]):
		return CAMERA_KEYS[0]
	for i in range(CAMERA_KEYS.size() - 1):
		var a: Dictionary = CAMERA_KEYS[i]
		var b: Dictionary = CAMERA_KEYS[i + 1]
		var ta := float(a.get("t", 0.0))
		var tb := float(b.get("t", 1.0))
		if t <= tb:
			var local_t := 0.0 if is_equal_approx(tb, ta) else clampf((t - ta) / (tb - ta), 0.0, 1.0)
			return {
				"offset": (a.get("offset", Vector3.ZERO) as Vector3).lerp(b.get("offset", Vector3.ZERO), local_t),
				"look": (a.get("look", Vector3.ZERO) as Vector3).lerp(b.get("look", Vector3.ZERO), local_t),
			}
	return CAMERA_KEYS[CAMERA_KEYS.size() - 1]

func _animate_scene(delta: float) -> void:
	var speed_scale := 0.5
	for i in range(_instance_nodes.size()):
		var node: Node3D = _instance_nodes[i]
		if node == null:
			continue
		node.rotate_y(delta * speed_scale * (1.0 + float(i % 5) * 0.1))

	hero_effect_asset.rotate_y(delta * 0.7)

	var light_phase := _elapsed_s * 0.5
	omni_light_a.global_position = _focus_point + Vector3(14.0 * sin(light_phase), 8.0, 14.0 * cos(light_phase))
	omni_light_b.global_position = _focus_point + Vector3(14.0 * cos(light_phase + PI * 0.5), 9.5, 14.0 * sin(light_phase + PI * 0.5))

	spot_light.global_position = _focus_point + Vector3(18.0 * sin(light_phase * 0.7), 12.0, 18.0 * cos(light_phase * 0.7))
	spot_light.look_at(_focus_point + Vector3(0.0, 2.5, 0.0), Vector3.UP)

	if _settings_apply_accum >= SETTINGS_APPLY_INTERVAL:
		_settings_apply_accum = 0.0
		_apply_dynamic_phase_settings()

func _sample_metrics(delta: float) -> void:
	if delta <= 0.0:
		return

	var frame_ms := delta * 1000.0
	var fps := 1.0 / delta
	_overall_frame_ms.append(frame_ms)
	_overall_fps.append(fps)

	if _phase_name.is_empty() or not _phase_runtime.has(_phase_name):
		return

	var phase_data: Dictionary = _phase_runtime[_phase_name]
	phase_data["frame_ms"].append(frame_ms)
	phase_data["fps"].append(fps)

	for key in MONITOR_KEYS:
		var monitor_id := "gaussian_splatting/%s" % key
		if not Performance.has_custom_monitor(monitor_id):
			continue
		var value := float(Performance.get_custom_monitor(monitor_id))
		if not _monitor_max.has(key) or value > float(_monitor_max[key]):
			_monitor_max[key] = value
		var phase_monitor_max: Dictionary = phase_data.get("monitor_max", {})
		if not phase_monitor_max.has(key) or value > float(phase_monitor_max[key]):
			phase_monitor_max[key] = value
		phase_data["monitor_max"] = phase_monitor_max

	var renderer = _get_primary_renderer()
	if renderer != null and renderer.has_method("get_render_stats"):
		var stats = renderer.get_render_stats()
		if stats is Dictionary:
			_last_renderer_stats = stats
			phase_data["renderer_stats"] = stats

	_phase_runtime[_phase_name] = phase_data

func _get_primary_renderer():
	if streaming_world != null and streaming_world.has_method("get_renderer"):
		var renderer = streaming_world.get_renderer()
		if renderer != null:
			return renderer
	if hero_effect_asset != null and hero_effect_asset.has_method("get_renderer"):
		return hero_effect_asset.get_renderer()
	return null

func _finish_benchmark() -> void:
	_state_running = false
	_restore_project_settings()
	_result_report = _build_report()
	_write_report(_result_report)

	if _orchestrated:
		emit_signal("benchmark_scene_finished", {
			"success": true,
			"lane_id": "unified_composite",
			"report_path": output_path,
			"report": _result_report,
		})
		queue_free()
		return

	if headless_summary or DisplayServer.get_name() == "headless":
		_print_headless_summary(_result_report)
		get_tree().quit(0)
		return

	performance_overlay.visible = false
	if results_panel.has_method("show_report"):
		results_panel.show_report(_result_report)
	else:
		results_panel.visible = true
	print("[BENCH] Benchmark complete. Press Esc to quit, R to rerun.")

func _build_report() -> Dictionary:
	var phase_summaries: Array = []
	for phase_variant in _phase_definitions:
		var phase: Dictionary = phase_variant
		var name := str(phase.get("name", "Phase"))
		var runtime: Dictionary = _phase_runtime.get(name, {})
		var summary: Dictionary = BenchmarkMetricsUtil.summarize_samples(runtime.get("frame_ms", []), runtime.get("fps", []))
		summary["name"] = name
		summary["start_s"] = float(phase.get("start_s", 0.0))
		summary["end_s"] = float(phase.get("end_s", 0.0))
		summary["monitor_max"] = runtime.get("monitor_max", {})
		phase_summaries.append(summary)

	var overall: Dictionary = BenchmarkMetricsUtil.summarize_samples(_overall_frame_ms, _overall_fps)
	overall["route_uid"] = _last_renderer_stats.get("route_uid", "")
	overall["sort_route_uid"] = _last_renderer_stats.get("sort_route_uid", "")
	overall["data_source"] = _last_renderer_stats.get("data_source", "")
	overall["stage_cull_status"] = _last_renderer_stats.get("stage_cull_status", "")
	overall["stage_sort_status"] = _last_renderer_stats.get("stage_sort_status", "")
	overall["stage_raster_status"] = _last_renderer_stats.get("stage_raster_status", "")

	var settings_now := _collect_current_project_settings()
	var score: float = BenchmarkMetricsUtil.compute_score(overall, _monitor_max)
	var recommendations: Array[Dictionary] = BenchmarkMetricsUtil.build_recommendations({
		"overall": overall,
		"monitor_max": _monitor_max,
		"project_settings": settings_now,
	})

	return {
		"name": "GodotGS Unified Benchmark",
		"duration_s": _elapsed_s,
		"configured_duration_s": benchmark_duration,
		"output_path": output_path,
		"timestamp_unix": Time.get_unix_time_from_system(),
		"platform": OS.get_name(),
		"phase_summaries": phase_summaries,
		"overall": overall,
		"monitor_max": _monitor_max,
		"project_settings": settings_now,
		"score": score,
		"recommendations": recommendations,
	}

func _write_report(report: Dictionary) -> void:
	var file := FileAccess.open(output_path, FileAccess.WRITE)
	if file == null:
		push_warning("[BENCH] Failed to write report: %s" % output_path)
		return
	file.store_string(JSON.stringify(report, "  "))
	print("[BENCH] Report written to %s" % output_path)

func _print_headless_summary(report: Dictionary) -> void:
	var overall: Dictionary = report.get("overall", {})
	print("[BENCH] Summary | score=%.1f avg_fps=%.1f p1_fps=%.1f p99_ms=%.2f" % [
		float(report.get("score", 0.0)),
		float(overall.get("avg_fps", 0.0)),
		float(overall.get("p1_fps", 0.0)),
		float(overall.get("p99_frame_ms", 0.0)),
	])
	var recommendations: Array = report.get("recommendations", [])
	for rec_variant in recommendations:
		if not (rec_variant is Dictionary):
			continue
		var rec: Dictionary = rec_variant
		print("[BENCH] Suggest %s: %s -> %s" % [
			str(rec.get("setting", "setting")),
			str(rec.get("current", "n/a")),
			str(rec.get("suggested", "n/a")),
		])

func _snapshot_project_settings() -> void:
	var ps := ProjectSettings
	for key in PROJECT_SETTING_KEYS:
		var existed := ps.has_setting(key)
		var value = null
		if existed:
			value = ps.get_setting(key)
		_settings_snapshot[key] = {
			"existed": existed,
			"value": value,
		}

func _apply_base_project_settings() -> void:
	_set_project_setting("rendering/gaussian_splatting/streaming/enabled", true)
	_set_project_setting("rendering/gaussian_splatting/instance_pipeline/enabled", true)
	_set_project_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", false)
	_set_project_setting("rendering/gaussian_splatting/lighting/direct_light_scale", 1.0)
	_set_project_setting("rendering/gaussian_splatting/lighting/indirect_sh_scale", 0.7)
	_set_project_setting("rendering/gaussian_splatting/lighting/shadow_strength", 0.5)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_strength", 0.0)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/lod/max_distance", 50.0)
	_set_project_setting("rendering/gaussian_splatting/lod/bias", 1.0)
	_set_project_setting("rendering/gaussian_splatting/lod/hysteresis_zone", 0.5)

func _restore_project_settings() -> void:
	if _settings_restored:
		return
	_settings_restored = true
	var ps := ProjectSettings
	for key in _settings_snapshot.keys():
		var snapshot: Dictionary = _settings_snapshot[key]
		if bool(snapshot.get("existed", false)):
			ps.set_setting(key, snapshot.get("value"))

func _collect_current_project_settings() -> Dictionary:
	var out := {}
	for key in PROJECT_SETTING_KEYS:
		if ProjectSettings.has_setting(key):
			out[key] = ProjectSettings.get_setting(key)
	return out

func _set_project_setting(key: String, value) -> void:
	ProjectSettings.set_setting(key, value)

func _set_world_lod_bias(value: float) -> void:
	if streaming_world != null:
		streaming_world.set("quality/lod_bias", value)
	for node_variant in _instance_nodes:
		var node: Node3D = node_variant
		if node != null:
			node.set("quality/lod_bias", value)
	if hero_effect_asset != null:
		hero_effect_asset.set("quality/lod_bias", value)

func _set_scene_visibility(show_streaming: bool, show_instances: bool, show_hero: bool) -> void:
	streaming_world.visible = show_streaming
	instance_root.visible = show_instances
	hero_effect_asset.visible = show_hero

func _apply_environment_profile(profile: String) -> void:
	if world_environment == null:
		return
	var env := world_environment.environment
	if env == null:
		return

	match profile:
		"neutral":
			env.adjustment_enabled = false
			env.tonemap_exposure = 1.0
			env.glow_enabled = false
		"showcase":
			env.adjustment_enabled = true
			env.adjustment_brightness = 1.02
			env.adjustment_contrast = 1.08
			env.adjustment_saturation = 1.12
			env.tonemap_exposure = 1.08
			env.glow_enabled = true
		"dramatic":
			env.adjustment_enabled = true
			env.adjustment_brightness = 0.95
			env.adjustment_contrast = 1.2
			env.adjustment_saturation = 1.2
			env.tonemap_exposure = 1.15
			env.glow_enabled = true

func _setup_phase_warmup() -> void:
	_set_scene_visibility(true, false, false)
	_apply_environment_profile("neutral")
	_set_world_lod_bias(0.75)
	directional_light.light_energy = 1.15
	omni_light_a.visible = false
	omni_light_b.visible = false
	spot_light.visible = false
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 0)

func _setup_phase_instance() -> void:
	_set_scene_visibility(false, true, true)
	_apply_environment_profile("neutral")
	_set_world_lod_bias(0.85)
	directional_light.light_energy = 1.0
	omni_light_a.visible = true
	omni_light_b.visible = true
	spot_light.visible = false
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 0)

func _setup_phase_streaming() -> void:
	_set_scene_visibility(true, false, false)
	_apply_environment_profile("neutral")
	_set_world_lod_bias(0.8)
	directional_light.light_energy = 1.05
	omni_light_a.visible = false
	omni_light_b.visible = false
	spot_light.visible = false
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 0)

func _setup_phase_lighting() -> void:
	_set_scene_visibility(true, true, false)
	_apply_environment_profile("showcase")
	_set_world_lod_bias(0.9)
	directional_light.light_energy = 1.4
	omni_light_a.visible = true
	omni_light_b.visible = true
	spot_light.visible = true
	_set_project_setting("rendering/gaussian_splatting/lighting/direct_light_scale", 1.2)
	_set_project_setting("rendering/gaussian_splatting/lighting/indirect_sh_scale", 0.9)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)

func _setup_phase_effects() -> void:
	_set_scene_visibility(true, true, true)
	_apply_environment_profile("dramatic")
	_set_world_lod_bias(0.95)
	directional_light.light_energy = 1.2
	omni_light_a.visible = true
	omni_light_b.visible = true
	spot_light.visible = true
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", true)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_strength", 1.0)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_frequency", 1.7)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 4)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", true)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_radius", 7.0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_strength", 1.35)

func _setup_phase_lod() -> void:
	_set_scene_visibility(true, false, true)
	_apply_environment_profile("showcase")
	_set_world_lod_bias(1.1)
	directional_light.light_energy = 1.05
	omni_light_a.visible = false
	omni_light_b.visible = true
	spot_light.visible = true
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", false)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 0)
	_set_project_setting("rendering/gaussian_splatting/lod/max_distance", 40.0)

func _setup_phase_finalize() -> void:
	_set_scene_visibility(true, true, true)
	_apply_environment_profile("neutral")
	_set_world_lod_bias(0.8)
	directional_light.light_energy = 1.0
	omni_light_a.visible = false
	omni_light_b.visible = false
	spot_light.visible = false

func _apply_dynamic_phase_settings() -> void:
	if _phase_name == "Effects":
		var wind_angle := _elapsed_s * 0.6
		_set_project_setting("rendering/gaussian_splatting/animation/wind_direction_x", cos(wind_angle))
		_set_project_setting("rendering/gaussian_splatting/animation/wind_direction_y", 0.0)
		_set_project_setting("rendering/gaussian_splatting/animation/wind_direction_z", sin(wind_angle))
		_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_x", _focus_point.x + sin(_elapsed_s * 1.5) * 8.0)
		_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_y", _focus_point.y + 2.5)
		_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_z", _focus_point.z + cos(_elapsed_s * 1.5) * 8.0)
	elif _phase_name == "LOD":
		var lod_bias := 0.95 + 0.45 * (0.5 + 0.5 * sin(_elapsed_s * 1.4))
		_set_world_lod_bias(lod_bias)
		_set_project_setting("rendering/gaussian_splatting/lod/bias", lod_bias)
	elif _phase_name == "Lighting":
		var env := world_environment.environment
		if env != null:
			env.tonemap_exposure = 1.04 + 0.08 * sin(_elapsed_s * 0.8)
