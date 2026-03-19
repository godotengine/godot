extends Node3D

const BenchmarkMetricsUtil = preload("res://scripts/benchmark_metrics.gd")
const BenchmarkVisualMetrics = preload("res://scripts/benchmark_visual_metrics.gd")

const DEFAULT_BENCHMARK_DURATION := 20.0
const DEFAULT_BENCHMARK_WARMUP := 3.0
const DEFAULT_OUTPUT_PATH := "user://benchmark_suite_lane_results.json"
const DEFAULT_ASSET_PATH := "res://tests/fixtures/test_splats.ply"
const DEFAULT_CAPTURE_PROGRESS_MARKERS := PackedFloat32Array([0.25, 0.5, 0.75])
const DEFAULT_VISUAL_SSIM_THRESHOLD := 0.95
const DEFAULT_VISUAL_PSNR_THRESHOLD := 30.0
const SETTINGS_APPLY_INTERVAL := 0.1
const INSTANCE_SINGLE_PASS_SETTING := "rendering/gaussian_splatting/instance_pipeline/true_single_pass_enabled"
const INSTANCE_BENCH_SERIAL_MULTI_ASSET_SETTING := "rendering/gaussian_splatting/instance_pipeline/benchmark_allow_serial_multi_asset"

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
	INSTANCE_SINGLE_PASS_SETTING,
	INSTANCE_BENCH_SERIAL_MULTI_ASSET_SETTING,
	"rendering/gaussian_splatting/quality/tier_preset",
	"rendering/gaussian_splatting/quality/tier_apply_pipeline_toggles",
	"rendering/gaussian_splatting/quality/tier_apply_streaming_budgets",
	"rendering/gaussian_splatting/gpu_sorting/debug_validate_prefix",
	"rendering/gaussian_splatting/gpu_sorting/enable_prefix_readback",
	"rendering/gaussian_splatting/gpu_sorting/profiling_preserve_gpu_timestamps",
	"rendering/gaussian_splatting/gpu_sorting/max_overlap_records",
	"rendering/gaussian_splatting/lod/splat_skip_enabled",
	"rendering/gaussian_splatting/lod/sh_reduction_enabled",
	"rendering/gaussian_splatting/lod/opacity_fade_enabled",
	"rendering/gaussian_splatting/cull/overflow_autotune_enabled",
	"rendering/gaussian_splatting/rasterization/low_pass_filter",
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

const RENDER_TELEMETRY_KEYS := [
	"instance_pipeline_execution_mode",
	"instance_pipeline_execution_path",
	"instance_pipeline_execution_reason",
	"instance_pipeline_true_single_pass_enabled",
	"instance_pipeline_instance_count",
	"effective_quality_preset",
	"effective_max_splats",
	"effective_lod_enabled",
	"effective_lod_bias",
	"effective_lod_max_distance",
	"effective_distance_cull_enabled",
	"effective_distance_cull_start",
	"effective_distance_cull_max_rate",
	"effective_tiny_splat_screen_radius",
	"effective_overflow_autotune_enabled",
	"sort_active_algorithm",
	"sort_switch_reason",
	"sort_sync_fallback_count",
	"instance_sort_sync_fallback_count",
	"tile_sort_sync_fallback_count",
	"sort_cached_fallback_count",
	"sort_identity_fallback_count",
	"sort_cull_order_fallback_count",
	"sort_total_route_fallback_count",
	"gpu_frame_time_ms",
	"gpu_tile_binning_time_ms",
	"gpu_tile_prefix_time_ms",
	"gpu_tile_raster_time_ms",
	"gpu_tile_resolve_time_ms",
	"gpu_timing_frame_serial",
	"gpu_timing_available",
	"gpu_frame_estimate_ms",
	"gpu_frame_time_source",
	"gpu_timeline_stall_count",
	"gpu_timeline_stall_ms",
	"gpu_timing_frames_behind",
	"gpu_timeline_inflight_frames",
	"gpu_pass_breakdown_available",
	"streaming_queue_pressure_active",
	"streaming_queue_pressure_frames",
	"streaming_vram_cap_hit_frames",
	"streaming_upload_bandwidth_cap_hit",
	"streaming_chunk_load_cap_hit",
	"streaming_vram_chunk_cap_hit",
	"stage_sort_reason",
	"stage_raster_reason",
	"stage_cull_time_ms",
	"stage_sort_time_ms",
	"stage_raster_time_ms",
	"stage_composite_time_ms",
	"overlap_records",
	"overlap_record_budget",
	"overlap_record_budget_effective",
	"overlap_record_budget_configured",
	"overlap_thinning_keep_ratio",
	"sorted_indices_blend_fallback_active",
	"sorted_indices_blend_fallback_reason",
]

@export var lane_id := "static_baseline"
@export var lane_name := "Static Baseline"
@export var lane_description := "Single-asset baseline lane."
@export var lane_preset := "static_baseline"
@export var placeholder_asset_path := DEFAULT_ASSET_PATH
@export var default_duration_s := DEFAULT_BENCHMARK_DURATION

@onready var camera: Camera3D = $Camera3D
@onready var performance_overlay: Control = $PerformanceOverlay
@onready var instance_root: Node3D = $InstanceRoot
@onready var directional_light: DirectionalLight3D = $DirectionalLight3D
@onready var omni_light_a: OmniLight3D = $OmniLightA
@onready var omni_light_b: OmniLight3D = $OmniLightB
@onready var spot_light: SpotLight3D = $SpotLight

var benchmark_duration := DEFAULT_BENCHMARK_DURATION
var benchmark_warmup := DEFAULT_BENCHMARK_WARMUP
var output_path := DEFAULT_OUTPUT_PATH
var headless_summary := false
var lane_tag := ""
var asset_override_path := ""
var instancing_mode := "auto"
var capture_dir := ""
var reference_dir := ""
var capture_tag := ""
var visual_ssim_threshold := DEFAULT_VISUAL_SSIM_THRESHOLD
var visual_psnr_threshold := DEFAULT_VISUAL_PSNR_THRESHOLD

var _elapsed_s := 0.0
var _frame_ms: Array = []
var _fps: Array = []
var _warmup_frame_ms: Array = []
var _warmup_fps: Array = []
var _steady_frame_ms: Array = []
var _steady_fps: Array = []
var _monitor_max: Dictionary = {}
var _warmup_monitor_max: Dictionary = {}
var _steady_monitor_max: Dictionary = {}
var _settings_snapshot: Dictionary = {}
var _settings_restored := false
var _settings_apply_accum := 0.0
var _result_report: Dictionary = {}
var _state_running := true
var _focus_point := Vector3.ZERO
var _primary_renderer_owner: Node3D = null
var _instance_nodes: Array = []
var _max_node_visible_splats := 0
var _max_total_visible_splats := 0
var _lane_config: Dictionary = {}
var _capture_targets: Array[Dictionary] = []
var _capture_records: Array[Dictionary] = []

func _ready() -> void:
	benchmark_duration = default_duration_s
	_parse_args()
	_setup_runtime_state()
	_initialize_capture_targets()
	_snapshot_project_settings()
	_lane_config = _config_for_preset(lane_preset)
	_apply_lane_project_settings(_lane_config)
	_build_instances(_lane_config)
	_apply_renderer_overrides_from_config()
	_focus_point = _resolve_focus_point()
	performance_overlay.visible = true
	print("[BENCH-LANE] start | lane=%s preset=%s duration=%.1fs output=%s asset=%s" % [
		lane_id,
		lane_preset,
		benchmark_duration,
		output_path,
		_resolved_asset_path(),
	])
	print("[BENCH-LANE] warmup | lane=%s warmup=%.1fs" % [lane_id, benchmark_warmup])

func _process(delta: float) -> void:
	if not _state_running:
		return
	if delta <= 0.0:
		return

	_elapsed_s += delta
	_settings_apply_accum += delta

	_update_camera()
	_animate_lane(delta)
	_sample_metrics(delta)

	if _settings_apply_accum >= SETTINGS_APPLY_INTERVAL:
		_settings_apply_accum = 0.0
		_apply_dynamic_lane_settings()

	_capture_due_frames(false)

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

func _parse_args() -> void:
	var args := OS.get_cmdline_args()
	for i in range(args.size()):
		var arg := str(args[i])
		if arg.begins_with("--benchmark-output="):
			output_path = arg.replace("--benchmark-output=", "")
		elif arg == "--benchmark-output" and i + 1 < args.size():
			output_path = str(args[i + 1])
		elif arg.begins_with("--benchmark-duration="):
			benchmark_duration = max(5.0, float(arg.replace("--benchmark-duration=", "")))
		elif arg == "--benchmark-duration" and i + 1 < args.size():
			benchmark_duration = max(5.0, float(args[i + 1]))
		elif arg.begins_with("--benchmark-warmup="):
			benchmark_warmup = max(0.0, float(arg.replace("--benchmark-warmup=", "")))
		elif arg == "--benchmark-warmup" and i + 1 < args.size():
			benchmark_warmup = max(0.0, float(args[i + 1]))
		elif arg == "--benchmark-headless-summary":
			headless_summary = true
		elif arg.begins_with("--benchmark-asset="):
			asset_override_path = arg.replace("--benchmark-asset=", "")
		elif arg == "--benchmark-asset" and i + 1 < args.size():
			asset_override_path = str(args[i + 1])
		elif arg.begins_with("--benchmark-lane-tag="):
			lane_tag = arg.replace("--benchmark-lane-tag=", "")
		elif arg == "--benchmark-lane-tag" and i + 1 < args.size():
			lane_tag = str(args[i + 1])
		elif arg.begins_with("--benchmark-instancing-mode="):
			instancing_mode = _normalize_instancing_mode(arg.replace("--benchmark-instancing-mode=", ""))
		elif arg == "--benchmark-instancing-mode" and i + 1 < args.size():
			instancing_mode = _normalize_instancing_mode(str(args[i + 1]))
		elif arg.begins_with("--benchmark-capture-dir="):
			capture_dir = arg.replace("--benchmark-capture-dir=", "")
		elif arg == "--benchmark-capture-dir" and i + 1 < args.size():
			capture_dir = str(args[i + 1])
		elif arg.begins_with("--benchmark-reference-dir="):
			reference_dir = arg.replace("--benchmark-reference-dir=", "")
		elif arg == "--benchmark-reference-dir" and i + 1 < args.size():
			reference_dir = str(args[i + 1])
		elif arg.begins_with("--benchmark-capture-tag="):
			capture_tag = arg.replace("--benchmark-capture-tag=", "")
		elif arg == "--benchmark-capture-tag" and i + 1 < args.size():
			capture_tag = str(args[i + 1])
		elif arg.begins_with("--benchmark-ssim-threshold="):
			visual_ssim_threshold = maxf(0.0, minf(1.0, float(arg.replace("--benchmark-ssim-threshold=", ""))))
		elif arg == "--benchmark-ssim-threshold" and i + 1 < args.size():
			visual_ssim_threshold = maxf(0.0, minf(1.0, float(args[i + 1])))
		elif arg.begins_with("--benchmark-psnr-threshold="):
			visual_psnr_threshold = maxf(0.0, float(arg.replace("--benchmark-psnr-threshold=", "")))
		elif arg == "--benchmark-psnr-threshold" and i + 1 < args.size():
			visual_psnr_threshold = maxf(0.0, float(args[i + 1]))

func _initialize_capture_targets() -> void:
	_capture_targets.clear()
	_capture_records.clear()
	if capture_dir.is_empty():
		return
	for i in range(DEFAULT_CAPTURE_PROGRESS_MARKERS.size()):
		_capture_targets.append({
			"index": i,
			"fraction": float(DEFAULT_CAPTURE_PROGRESS_MARKERS[i]),
			"captured": false,
		})

func _normalize_instancing_mode(value: String) -> String:
	match value.strip_edges().to_lower():
		"serial":
			return "serial"
		"single_pass", "single-pass":
			return "single_pass"
		_:
			return "auto"

func _setup_runtime_state() -> void:
	OS.set_low_processor_usage_mode(false)
	OS.set_low_processor_usage_mode_sleep_usec(0)
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

func _config_for_preset(preset: String) -> Dictionary:
	match preset:
		"streaming_corridor":
			return {
				"camera_mode": "corridor",
				"cols": 2,
				"rows": 8,
				"spacing": 14.0,
				"max_splats": 120000,
				"lod_bias": 1.15,
				"lod_max_distance": 500.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": false,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"corridor_length": 120.0,
				"corridor_speed": 14.0,
			}
		"city_flyover":
			return {
				"camera_mode": "flyover",
				"cols": 4,
				"rows": 4,
				"spacing": 18.0,
				"max_splats": 140000,
				"lod_bias": 1.1,
				"lod_max_distance": 650.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": false,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"orbit_radius": 90.0,
				"orbit_height": 38.0,
				"orbit_speed": 0.18,
			}
		"instance_storm":
			return {
				"camera_mode": "orbit",
				"cols": 6,
				"rows": 6,
				"spacing": 10.0,
				"max_splats": 100000,
				"lod_bias": 1.2,
				"lod_max_distance": 450.0,
				"rotate_instances": true,
				"rotate_speed": 0.8,
				"lighting_animate": false,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"orbit_radius": 48.0,
				"orbit_height": 14.0,
				"orbit_speed": 0.32,
			}
		"lighting_stress":
			return {
				"camera_mode": "orbit",
				"cols": 3,
				"rows": 3,
				"spacing": 16.0,
				"max_splats": 75000,
				"lod_bias": 1.1,
				"lod_max_distance": 500.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": true,
				"direct_light_scale": 1.0,
				"indirect_sh_scale": 0.75,
				"shadow_strength": 0.5,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"orbit_radius": 42.0,
				"orbit_height": 12.0,
				"orbit_speed": 0.25,
			}
		"animation_arena":
			return {
				"camera_mode": "orbit",
				"cols": 3,
				"rows": 3,
				"spacing": 12.0,
				"max_splats": 90000,
				"lod_bias": 1.15,
				"lod_max_distance": 450.0,
				"rotate_instances": true,
				"rotate_speed": 0.5,
				"lighting_animate": true,
				"direct_light_scale": 1.0,
				"indirect_sh_scale": 0.75,
				"shadow_strength": 0.5,
				"wind_enabled": true,
				"wind_strength": 0.45,
				"effects_enabled": true,
				"orbit_radius": 34.0,
				"orbit_height": 10.0,
				"orbit_speed": 0.35,
			}
		"lod_torture":
			return {
				"camera_mode": "lod_pulse",
				"cols": 2,
				"rows": 4,
				"spacing": 18.0,
				"max_splats": 130000,
				"lod_bias": 1.0,
				"lod_max_distance": 550.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": false,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"pulse_min_radius": 20.0,
				"pulse_max_radius": 130.0,
				"pulse_speed": 1.25,
			}
		"integrity_sentinel":
			return {
				"camera_mode": "sentinel",
				"cols": 1,
				"rows": 1,
				"spacing": 1.0,
				"max_splats": 50000,
				"lod_bias": 1.1,
				"lod_max_distance": 350.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": false,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
			}
		"long_soak":
			return {
				"camera_mode": "orbit",
				"cols": 2,
				"rows": 2,
				"spacing": 24.0,
				"max_splats": 70000,
				"lod_bias": 1.2,
				"lod_max_distance": 600.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": true,
				"direct_light_scale": 1.0,
				"indirect_sh_scale": 0.75,
				"shadow_strength": 0.5,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"orbit_radius": 58.0,
				"orbit_height": 18.0,
				"orbit_speed": 0.08,
			}
		"parity_fidelity":
			return {
				"camera_mode": "orbit",
				"cols": 1,
				"rows": 1,
				"spacing": 1.0,
				"max_splats": 10000000,
				"lod_bias": 1.0,
				"lod_max_distance": 1500.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": false,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"orbit_radius": 26.0,
				"orbit_height": 9.0,
				"orbit_speed": 0.14,
				"splat_skip_enabled": false,
				"sh_reduction_enabled": false,
				"opacity_fade_enabled": false,
				"distance_cull_enabled": false,
				"distance_cull_start": 1000.0,
				"distance_cull_max_rate": 0.0,
				"tiny_splat_screen_radius": 0.1,
				"overflow_autotune_enabled": false,
				"max_overlap_records": 200000000,
			}
		_:
			return {
				"camera_mode": "orbit",
				"cols": 1,
				"rows": 1,
				"spacing": 1.0,
				"max_splats": 50000,
				"lod_bias": 1.2,
				"lod_max_distance": 500.0,
				"rotate_instances": false,
				"rotate_speed": 0.0,
				"lighting_animate": false,
				"wind_enabled": false,
				"wind_strength": 0.0,
				"effects_enabled": false,
				"orbit_radius": 34.0,
				"orbit_height": 11.0,
				"orbit_speed": 0.24,
			}

func _resolved_asset_path() -> String:
	if not asset_override_path.is_empty():
		return asset_override_path
	if not placeholder_asset_path.is_empty():
		return placeholder_asset_path
	return DEFAULT_ASSET_PATH

func _build_instances(config: Dictionary) -> void:
	for child in instance_root.get_children():
		child.queue_free()
	_instance_nodes.clear()
	_primary_renderer_owner = null
	_max_node_visible_splats = 0
	_max_total_visible_splats = 0

	var cols := int(config.get("cols", 1))
	var rows := int(config.get("rows", 1))
	var spacing := float(config.get("spacing", 1.0))
	var max_splats := int(config.get("max_splats", 50000))
	var lod_bias := float(config.get("lod_bias", 1.1))
	var max_render_distance := float(config.get("lod_max_distance", 400.0))
	var asset_path := _resolved_asset_path()

	for row in range(rows):
		for col in range(cols):
			var node := GaussianSplatNode3D.new()
			node.name = "Lane_%02d_%02d" % [row, col]
			node.ply_file_path = asset_path
			node.position = Vector3(
				(float(col) - float(cols - 1) * 0.5) * spacing,
				0.0,
				(float(row) - float(rows - 1) * 0.5) * spacing
			)
			node.set("quality/preset", GaussianSplatNode3D.QUALITY_CUSTOM)
			node.set("quality/max_splat_count", max_splats)
			node.set("quality/lod_bias", lod_bias)
			node.set("quality/max_render_distance", max_render_distance)
			instance_root.add_child(node)
			_instance_nodes.append(node)
			if _primary_renderer_owner == null:
				_primary_renderer_owner = node

func _resolve_focus_point() -> Vector3:
	return instance_root.global_position + Vector3(0.0, 2.0, 0.0)

func _update_camera() -> void:
	var camera_mode := str(_lane_config.get("camera_mode", "orbit"))

	if camera_mode == "corridor":
		var length := float(_lane_config.get("corridor_length", 80.0))
		var speed := float(_lane_config.get("corridor_speed", 10.0))
		var z := -length + fposmod(_elapsed_s * speed, length * 2.0)
		var x := sin(_elapsed_s * 0.45) * 8.0
		camera.global_position = _focus_point + Vector3(x, 9.0, z)
		camera.look_at(_focus_point + Vector3(0.0, 0.0, z + 25.0), Vector3.UP)
		return

	if camera_mode == "flyover":
		var fly_radius := float(_lane_config.get("orbit_radius", 85.0))
		var fly_height := float(_lane_config.get("orbit_height", 36.0))
		var fly_speed := float(_lane_config.get("orbit_speed", 0.2))
		var fly_angle := _elapsed_s * fly_speed
		camera.global_position = _focus_point + Vector3(cos(fly_angle) * fly_radius, fly_height, sin(fly_angle) * fly_radius)
		camera.look_at(_focus_point + Vector3(0.0, -2.0, 0.0), Vector3.UP)
		return

	if camera_mode == "lod_pulse":
		var pulse_min := float(_lane_config.get("pulse_min_radius", 20.0))
		var pulse_max := float(_lane_config.get("pulse_max_radius", 100.0))
		var pulse_speed := float(_lane_config.get("pulse_speed", 1.0))
		var pulse := 0.5 + 0.5 * sin(_elapsed_s * pulse_speed)
		var radius := lerpf(pulse_min, pulse_max, pulse)
		var pulse_angle := _elapsed_s * 0.4
		camera.global_position = _focus_point + Vector3(cos(pulse_angle) * radius, 10.0, sin(pulse_angle) * radius)
		camera.look_at(_focus_point, Vector3.UP)
		return

	if camera_mode == "sentinel":
		var t := fposmod(_elapsed_s, 6.0)
		if t < 2.0:
			camera.global_position = _focus_point + Vector3(0.0, 11.0, 32.0)
		elif t < 4.0:
			camera.global_position = _focus_point + Vector3(22.0, 9.0, 10.0)
		else:
			camera.global_position = _focus_point + Vector3(-18.0, 13.0, -16.0)
		camera.look_at(_focus_point, Vector3.UP)
		return

	var orbit_radius := float(_lane_config.get("orbit_radius", 32.0))
	var orbit_height := float(_lane_config.get("orbit_height", 11.0))
	var orbit_speed := float(_lane_config.get("orbit_speed", 0.22))
	var angle := _elapsed_s * orbit_speed
	camera.global_position = _focus_point + Vector3(cos(angle) * orbit_radius, orbit_height, sin(angle) * orbit_radius)
	camera.look_at(_focus_point, Vector3.UP)

func _animate_lane(delta: float) -> void:
	if bool(_lane_config.get("rotate_instances", false)):
		var rotate_speed := float(_lane_config.get("rotate_speed", 0.4))
		for i in range(_instance_nodes.size()):
			var node: Node3D = _instance_nodes[i]
			if node == null:
				continue
			node.rotate_y(delta * rotate_speed * (1.0 + float(i % 5) * 0.1))

	if bool(_lane_config.get("lighting_animate", false)):
		var phase := _elapsed_s * 0.6
		omni_light_a.global_position = _focus_point + Vector3(16.0 * sin(phase), 8.0, 16.0 * cos(phase))
		omni_light_b.global_position = _focus_point + Vector3(16.0 * cos(phase + PI * 0.5), 9.5, 16.0 * sin(phase + PI * 0.5))
		spot_light.global_position = _focus_point + Vector3(20.0 * sin(phase * 0.7), 12.0, 20.0 * cos(phase * 0.7))
		spot_light.look_at(_focus_point + Vector3(0.0, 2.0, 0.0), Vector3.UP)

func _apply_dynamic_lane_settings() -> void:
	if bool(_lane_config.get("effects_enabled", false)):
		var t := _elapsed_s * 0.9
		_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_x", _focus_point.x + sin(t) * 12.0)
		_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_y", _focus_point.y + 2.0)
		_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_z", _focus_point.z + cos(t) * 12.0)

	if str(_lane_config.get("camera_mode", "")) == "lod_pulse":
		var bias := 1.0 + 0.25 * (0.5 + 0.5 * sin(_elapsed_s * 1.2))
		_set_project_setting("rendering/gaussian_splatting/lod/bias", bias)
		for node_variant in _instance_nodes:
			var node: Node3D = node_variant
			if node != null:
				node.set("quality/lod_bias", bias)

	_apply_renderer_overrides_from_config()

func _apply_renderer_overrides_from_config() -> void:
	var has_distance_cull_enabled := _lane_config.has("distance_cull_enabled")
	var has_distance_cull_start := _lane_config.has("distance_cull_start")
	var has_distance_cull_max_rate := _lane_config.has("distance_cull_max_rate")
	var has_tiny_radius := _lane_config.has("tiny_splat_screen_radius")
	var has_overflow_autotune := _lane_config.has("overflow_autotune_enabled")
	if not has_distance_cull_enabled and not has_distance_cull_start and not has_distance_cull_max_rate and not has_tiny_radius and not has_overflow_autotune:
		return

	for node_variant in _instance_nodes:
		var node: Node3D = node_variant
		if node == null or not node.has_method("get_renderer"):
			continue
		var renderer = node.get_renderer()
		if renderer == null:
			continue
		if has_distance_cull_enabled and renderer.has_method("set_distance_cull_enabled"):
			renderer.set_distance_cull_enabled(bool(_lane_config.get("distance_cull_enabled")))
		if has_distance_cull_start and renderer.has_method("set_distance_cull_start"):
			renderer.set_distance_cull_start(float(_lane_config.get("distance_cull_start")))
		if has_distance_cull_max_rate and renderer.has_method("set_distance_cull_max_rate"):
			renderer.set_distance_cull_max_rate(float(_lane_config.get("distance_cull_max_rate")))
		if has_tiny_radius and renderer.has_method("set_tiny_splat_screen_radius"):
			renderer.set_tiny_splat_screen_radius(float(_lane_config.get("tiny_splat_screen_radius")))
		if has_overflow_autotune and renderer.has_method("set_overflow_autotune_enabled"):
			renderer.set_overflow_autotune_enabled(bool(_lane_config.get("overflow_autotune_enabled")))

func _sample_metrics(delta: float) -> void:
	var frame_ms := delta * 1000.0
	var fps := 1.0 / delta
	_frame_ms.append(frame_ms)
	_fps.append(fps)
	if _elapsed_s < benchmark_warmup:
		_warmup_frame_ms.append(frame_ms)
		_warmup_fps.append(fps)
	else:
		_steady_frame_ms.append(frame_ms)
		_steady_fps.append(fps)

	for key in MONITOR_KEYS:
		var monitor_id := "gaussian_splatting/%s" % key
		if not Performance.has_custom_monitor(monitor_id):
			continue
		var value := float(Performance.get_custom_monitor(monitor_id))
		_update_monitor_peak(_monitor_max, key, value)
		if _elapsed_s < benchmark_warmup:
			_update_monitor_peak(_warmup_monitor_max, key, value)
		else:
			_update_monitor_peak(_steady_monitor_max, key, value)

	var total_visible := 0
	for node_variant in _instance_nodes:
		var node: Node3D = node_variant
		if node != null and node.has_method("get_visible_splat_count"):
			total_visible += int(node.get_visible_splat_count())
	if total_visible > _max_total_visible_splats:
		_max_total_visible_splats = total_visible

	if _primary_renderer_owner != null and _primary_renderer_owner.has_method("get_visible_splat_count"):
		var visible := int(_primary_renderer_owner.get_visible_splat_count())
		if visible > _max_node_visible_splats:
			_max_node_visible_splats = visible

func _update_monitor_peak(target: Dictionary, key: String, value: float) -> void:
	if not target.has(key) or value > float(target[key]):
		target[key] = value

func _get_primary_renderer():
	if _primary_renderer_owner != null and _primary_renderer_owner.has_method("get_renderer"):
		return _primary_renderer_owner.get_renderer()
	return null

func _collect_primary_node_quality() -> Dictionary:
	var out := {}
	if _primary_renderer_owner == null:
		return out
	out["quality_preset"] = int(_primary_renderer_owner.get("quality/preset"))
	out["quality_lod_bias"] = float(_primary_renderer_owner.get("quality/lod_bias"))
	out["quality_max_splat_count"] = int(_primary_renderer_owner.get("quality/max_splat_count"))
	out["quality_max_render_distance"] = float(_primary_renderer_owner.get("quality/max_render_distance"))
	return out

func _finish_benchmark() -> void:
	_state_running = false
	_capture_due_frames(true)
	_result_report = _build_report()
	_restore_project_settings()
	_write_report(_result_report)

	if headless_summary or DisplayServer.get_name() == "headless":
		_print_headless_summary(_result_report)
		get_tree().quit(0)
		return

	print("[BENCH-LANE] complete | lane=%s press Esc to quit, R to rerun." % lane_id)

func _build_report() -> Dictionary:
	var overall: Dictionary = BenchmarkMetricsUtil.summarize_samples(_frame_ms, _fps)
	var warmup_overall: Dictionary = BenchmarkMetricsUtil.summarize_samples(_warmup_frame_ms, _warmup_fps)
	var steady_overall: Dictionary = BenchmarkMetricsUtil.summarize_samples(_steady_frame_ms, _steady_fps)
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
	if renderer_stats.has("gpu_frame_time_ms"):
		var gpu_frame_ms := float(renderer_stats.get("gpu_frame_time_ms", 0.0))
		var gpu_frame_source := str(renderer_stats.get("gpu_frame_time_source", "unavailable"))
		var gpu_frame_estimate := float(renderer_stats.get("gpu_frame_estimate_ms", 0.0))
		if gpu_frame_ms <= 0.0 and gpu_frame_source == "stage_estimate":
			gpu_frame_ms = gpu_frame_estimate
		overall["gpu_time_frame_ms"] = gpu_frame_ms
		overall["gpu_time_frame_estimate_ms"] = gpu_frame_estimate
		overall["gpu_time_frame_source"] = gpu_frame_source
		overall["gpu_timing_available"] = bool(renderer_stats.get("gpu_timing_available", false))
	if renderer_stats.has("stage_cull_time_ms"):
		overall["gpu_time_cull_ms"] = float(renderer_stats.get("stage_cull_time_ms", 0.0))
	if renderer_stats.has("gpu_tile_raster_time_ms"):
		overall["gpu_time_raster_ms"] = float(renderer_stats.get("gpu_tile_raster_time_ms", 0.0))
	if renderer_stats.has("instance_pipeline_execution_mode"):
		overall["instance_pipeline_execution_mode"] = renderer_stats.get("instance_pipeline_execution_mode", "")
	if renderer_stats.has("instance_pipeline_execution_path"):
		overall["instance_pipeline_execution_path"] = renderer_stats.get("instance_pipeline_execution_path", "")
	if renderer_stats.has("instance_pipeline_execution_reason"):
		overall["instance_pipeline_execution_reason"] = renderer_stats.get("instance_pipeline_execution_reason", "")
	var renderer_telemetry: Dictionary = {}
	for key_variant in RENDER_TELEMETRY_KEYS:
		var key := str(key_variant)
		if renderer_stats.has(key):
			var value = renderer_stats.get(key)
			overall[key] = value
			renderer_telemetry[key] = value

	var score_summary: Dictionary = overall
	var score_source := "overall"
	var score_monitor_max: Dictionary = _monitor_max
	if BenchmarkMetricsUtil.has_samples(steady_overall):
		score_summary = steady_overall
		score_source = "steady_overall"
		score_monitor_max = _steady_monitor_max

	var settings_now := _collect_current_project_settings()
	var score: float = BenchmarkMetricsUtil.compute_score(score_summary, score_monitor_max)
	var warmup_score = null
	if BenchmarkMetricsUtil.has_samples(warmup_overall):
		warmup_score = BenchmarkMetricsUtil.compute_score(warmup_overall, _warmup_monitor_max)
	var steady_score = null
	if BenchmarkMetricsUtil.has_samples(steady_overall):
		steady_score = BenchmarkMetricsUtil.compute_score(steady_overall, _steady_monitor_max)
	var recommendations: Array[Dictionary] = BenchmarkMetricsUtil.build_recommendations({
		"overall": score_summary,
		"monitor_max": score_monitor_max,
		"project_settings": settings_now,
	})
	var visual_summary := _build_visual_summary()

	return {
		"name": "GodotGS Benchmark Lane",
		"scene": get_tree().current_scene.scene_file_path,
		"lane_id": lane_id,
		"lane_name": lane_name,
		"lane_description": lane_description,
		"lane_preset": lane_preset,
		"lane_tag": lane_tag,
		"capture_tag": _resolved_capture_tag(),
		"instancing_mode": instancing_mode,
		"instancing_execution_mode": overall.get("instance_pipeline_execution_mode", ""),
		"instancing_execution_path": overall.get("instance_pipeline_execution_path", ""),
		"instancing_execution_reason": overall.get("instance_pipeline_execution_reason", ""),
		"duration_s": _elapsed_s,
		"configured_duration_s": benchmark_duration,
		"warmup_duration_s": benchmark_warmup,
		"output_path": output_path,
		"capture_output_dir": capture_dir,
		"capture_reference_dir": reference_dir,
		"timestamp_unix": Time.get_unix_time_from_system(),
		"platform": OS.get_name(),
		"asset_path": _resolved_asset_path(),
		"instance_count": _instance_nodes.size(),
		"node_visible_splats_max": _max_total_visible_splats,
		"node_total_visible_splats_max": _max_total_visible_splats,
		"node_primary_visible_splats_max": _max_node_visible_splats,
		"warmup_frame_count": int(warmup_overall.get("sample_count", 0)),
		"steady_frame_count": int(steady_overall.get("sample_count", 0)),
		"lane_config": _lane_config,
		"primary_node_quality": _collect_primary_node_quality(),
		"overall": overall,
		"warmup_overall": warmup_overall,
		"steady_overall": steady_overall,
		"renderer_telemetry": renderer_telemetry,
		"score_source": score_source,
		"monitor_max": _monitor_max,
		"warmup_monitor_max": _warmup_monitor_max,
		"steady_monitor_max": _steady_monitor_max,
		"score_monitor_max": score_monitor_max,
		"project_settings": settings_now,
		"score": score,
		"warmup_score": warmup_score,
		"steady_score": steady_score,
		"captures": _capture_records,
		"visual_summary": visual_summary,
		"recommendations": recommendations,
	}

func _write_report(report: Dictionary) -> void:
	var file := FileAccess.open(output_path, FileAccess.WRITE)
	if file == null:
		push_warning("[BENCH-LANE] Failed to write report: %s" % output_path)
		return
	file.store_string(JSON.stringify(report, "  "))
	print("[BENCH-LANE] report written to %s" % output_path)

func _print_headless_summary(report: Dictionary) -> void:
	var overall: Dictionary = report.get("overall", {})
	var steady: Dictionary = report.get("steady_overall", {})
	print("[BENCH-LANE] Summary | lane=%s mode=%s score=%.1f avg_fps=%.1f p1_fps=%.1f steady_p1_fps=%.1f p99_ms=%.2f gpu_ms=%.3f gpu_src=%s warmup_frames=%d steady_frames=%d visible_max=%d" % [
		lane_id,
		instancing_mode,
		float(report.get("score", 0.0)),
		float(overall.get("avg_fps", 0.0)),
		float(overall.get("p1_fps", 0.0)),
		float(steady.get("p1_fps", 0.0)),
		float(overall.get("p99_frame_ms", 0.0)),
		float(overall.get("gpu_time_frame_ms", 0.0)),
		str(overall.get("gpu_time_frame_source", "unavailable")),
		int(report.get("warmup_frame_count", 0)),
		int(report.get("steady_frame_count", 0)),
		int(report.get("node_visible_splats_max", 0)),
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

func _setting_snapshot_value_or_default(key: String, fallback):
	if _settings_snapshot.has(key):
		var snapshot: Dictionary = _settings_snapshot[key]
		if bool(snapshot.get("existed", false)):
			return snapshot.get("value")
	return fallback

func _apply_lane_project_settings(config: Dictionary) -> void:
	_set_project_setting("rendering/gaussian_splatting/streaming/enabled", true)
	_set_project_setting("rendering/gaussian_splatting/instance_pipeline/enabled", true)
	var bench_serial_multi_asset_default := bool(_setting_snapshot_value_or_default(INSTANCE_BENCH_SERIAL_MULTI_ASSET_SETTING, false))
	_set_project_setting(INSTANCE_BENCH_SERIAL_MULTI_ASSET_SETTING, bench_serial_multi_asset_default)
	if instancing_mode == "serial":
		_set_project_setting(INSTANCE_SINGLE_PASS_SETTING, false)
		_set_project_setting(INSTANCE_BENCH_SERIAL_MULTI_ASSET_SETTING, true)
	elif instancing_mode == "single_pass":
		_set_project_setting(INSTANCE_SINGLE_PASS_SETTING, true)
		_set_project_setting(INSTANCE_BENCH_SERIAL_MULTI_ASSET_SETTING, false)
	_set_project_setting("rendering/gaussian_splatting/quality/tier_preset", "custom")
	_set_project_setting("rendering/gaussian_splatting/quality/tier_apply_pipeline_toggles", false)
	_set_project_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", false)
	_set_project_setting("rendering/gaussian_splatting/gpu_sorting/debug_validate_prefix", false)
	_set_project_setting("rendering/gaussian_splatting/gpu_sorting/enable_prefix_readback", false)
	_set_project_setting("rendering/gaussian_splatting/gpu_sorting/profiling_preserve_gpu_timestamps", true)
	var overlap_budget_default := int(_setting_snapshot_value_or_default("rendering/gaussian_splatting/gpu_sorting/max_overlap_records", 100000000))
	_set_project_setting("rendering/gaussian_splatting/gpu_sorting/max_overlap_records", int(config.get("max_overlap_records", overlap_budget_default)))
	var splat_skip_default := bool(_setting_snapshot_value_or_default("rendering/gaussian_splatting/lod/splat_skip_enabled", true))
	var sh_reduction_default := bool(_setting_snapshot_value_or_default("rendering/gaussian_splatting/lod/sh_reduction_enabled", true))
	var opacity_fade_default := bool(_setting_snapshot_value_or_default("rendering/gaussian_splatting/lod/opacity_fade_enabled", true))
	_set_project_setting("rendering/gaussian_splatting/lod/splat_skip_enabled", bool(config.get("splat_skip_enabled", splat_skip_default)))
	_set_project_setting("rendering/gaussian_splatting/lod/sh_reduction_enabled", bool(config.get("sh_reduction_enabled", sh_reduction_default)))
	_set_project_setting("rendering/gaussian_splatting/lod/opacity_fade_enabled", bool(config.get("opacity_fade_enabled", opacity_fade_default)))
	_set_project_setting("rendering/gaussian_splatting/streaming/vram_budget_mb", 1024)
	_set_project_setting("rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame", 16)
	_set_project_setting("rendering/gaussian_splatting/streaming/pack_worker_threads", 4)
	_set_project_setting("rendering/gaussian_splatting/lod/max_distance", float(config.get("lod_max_distance", 500.0)))
	_set_project_setting("rendering/gaussian_splatting/lod/bias", float(config.get("lod_bias", 1.1)))
	_set_project_setting("rendering/gaussian_splatting/lod/hysteresis_zone", 0.6)
	var direct_light_default := float(_setting_snapshot_value_or_default("rendering/gaussian_splatting/lighting/direct_light_scale", 1.0))
	var indirect_sh_default := float(_setting_snapshot_value_or_default("rendering/gaussian_splatting/lighting/indirect_sh_scale", 0.0))
	var shadow_strength_default := float(_setting_snapshot_value_or_default("rendering/gaussian_splatting/lighting/shadow_strength", 0.0))
	_set_project_setting("rendering/gaussian_splatting/lighting/direct_light_scale", float(config.get("direct_light_scale", direct_light_default)))
	_set_project_setting("rendering/gaussian_splatting/lighting/indirect_sh_scale", float(config.get("indirect_sh_scale", indirect_sh_default)))
	_set_project_setting("rendering/gaussian_splatting/lighting/shadow_strength", float(config.get("shadow_strength", shadow_strength_default)))
	_set_project_setting("rendering/gaussian_splatting/animation/wind_enabled", bool(config.get("wind_enabled", false)))
	_set_project_setting("rendering/gaussian_splatting/animation/wind_strength", float(config.get("wind_strength", 0.0)))
	_set_project_setting("rendering/gaussian_splatting/animation/wind_frequency", 0.85)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_direction_x", 1.0)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_direction_y", 0.0)
	_set_project_setting("rendering/gaussian_splatting/animation/wind_direction_z", 0.2)
	_set_project_setting("rendering/gaussian_splatting/effects/max_effectors", 1 if bool(config.get("effects_enabled", false)) else 0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_enabled", bool(config.get("effects_enabled", false)))
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_radius", 12.0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_strength", 1.0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_falloff", 1.0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_frequency", 0.75)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_x", 0.0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_y", 2.0)
	_set_project_setting("rendering/gaussian_splatting/effects/sphere_effector_center_z", 0.0)

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

func _capture_due_frames(force_all: bool) -> void:
	if capture_dir.is_empty():
		return
	var steady_window := maxf(benchmark_duration - benchmark_warmup, 0.001)
	var steady_progress := 1.0 if force_all else clampf((_elapsed_s - benchmark_warmup) / steady_window, 0.0, 1.0)
	for i in range(_capture_targets.size()):
		var target: Dictionary = _capture_targets[i]
		if bool(target.get("captured", false)):
			continue
		var fraction := float(target.get("fraction", 1.0))
		if not force_all and (_elapsed_s < benchmark_warmup or steady_progress < fraction):
			continue
		target["captured"] = true
		_capture_targets[i] = target
		_capture_records.append(_capture_frame(int(target.get("index", i)), fraction))

func _capture_frame(slot_index: int, fraction: float) -> Dictionary:
	var capture_id := "capture_%02d" % [slot_index + 1]
	var basename := "%s__%s__%s.png" % [lane_id, _resolved_capture_tag(), capture_id]
	var capture_path := capture_dir.path_join(basename)
	var image := BenchmarkVisualMetrics.capture_viewport(get_viewport())
	var record := {
		"capture_id": capture_id,
		"capture_index": slot_index + 1,
		"capture_fraction": fraction,
		"capture_elapsed_s": _elapsed_s,
		"capture_path": capture_path,
		"reference_path": "",
		"saved": false,
		"reference_matched": false,
		"ssim": null,
		"psnr": null,
		"threshold_pass": null,
	}
	if image == null:
		record["capture_error"] = "viewport image unavailable"
		return record

	record["image_width"] = image.get_width()
	record["image_height"] = image.get_height()
	var save_error := BenchmarkVisualMetrics.save_png(image, capture_path)
	if save_error != OK:
		record["capture_error"] = "save_png failed (%s)" % [error_string(save_error)]
		return record
	record["saved"] = true

	var reference_path := _resolve_reference_path(capture_id)
	if reference_path.is_empty():
		return record
	record["reference_path"] = reference_path
	var reference_image := BenchmarkVisualMetrics.load_image(reference_path)
	if reference_image == null:
		record["comparison_error"] = "reference image could not be loaded"
		return record

	record["reference_matched"] = true
	var ssim := BenchmarkVisualMetrics.calculate_ssim(reference_image, image)
	var psnr := BenchmarkVisualMetrics.calculate_psnr(reference_image, image)
	record["ssim"] = ssim
	record["psnr"] = psnr
	record["threshold_pass"] = ssim >= visual_ssim_threshold and psnr >= visual_psnr_threshold
	return record

func _resolved_capture_tag() -> String:
	var tag := capture_tag
	if tag.is_empty():
		tag = lane_tag
	if tag.is_empty():
		tag = "default"
	return tag.replace(" ", "_").to_lower()

func _resolve_reference_path(capture_id: String) -> String:
	if reference_dir.is_empty():
		return ""
	var tagged_path := reference_dir.path_join("%s__%s__%s.png" % [lane_id, _resolved_capture_tag(), capture_id])
	if FileAccess.file_exists(tagged_path):
		return tagged_path
	var lane_default_path := reference_dir.path_join("%s__%s.png" % [lane_id, capture_id])
	if FileAccess.file_exists(lane_default_path):
		return lane_default_path
	return ""

func _build_visual_summary() -> Dictionary:
	var summary := {
		"capture_count": _capture_records.size(),
		"saved_capture_count": 0,
		"reference_match_count": 0,
		"missing_reference_count": 0,
		"capture_error_count": 0,
		"threshold_pass_count": 0,
		"ssim_threshold": visual_ssim_threshold,
		"psnr_threshold": visual_psnr_threshold,
		"ssim_min": null,
		"ssim_avg": null,
		"psnr_min": null,
		"psnr_avg": null,
	}
	var ssim_total := 0.0
	var ssim_count := 0
	var psnr_total := 0.0
	var psnr_count := 0
	for capture_variant in _capture_records:
		var capture: Dictionary = capture_variant
		if bool(capture.get("saved", false)):
			summary["saved_capture_count"] = int(summary["saved_capture_count"]) + 1
		else:
			summary["capture_error_count"] = int(summary["capture_error_count"]) + 1
		if bool(capture.get("reference_matched", false)):
			summary["reference_match_count"] = int(summary["reference_match_count"]) + 1
		elif bool(capture.get("saved", false)) and not reference_dir.is_empty():
			summary["missing_reference_count"] = int(summary["missing_reference_count"]) + 1

		var ssim_value = capture.get("ssim", null)
		if ssim_value is float or ssim_value is int:
			var ssim := float(ssim_value)
			ssim_total += ssim
			ssim_count += 1
			if summary["ssim_min"] == null or ssim < float(summary["ssim_min"]):
				summary["ssim_min"] = ssim

		var psnr_value = capture.get("psnr", null)
		if psnr_value is float or psnr_value is int:
			var psnr := float(psnr_value)
			psnr_total += psnr
			psnr_count += 1
			if summary["psnr_min"] == null or psnr < float(summary["psnr_min"]):
				summary["psnr_min"] = psnr

		if bool(capture.get("threshold_pass", false)):
			summary["threshold_pass_count"] = int(summary["threshold_pass_count"]) + 1

	if ssim_count > 0:
		summary["ssim_avg"] = ssim_total / float(ssim_count)
	if psnr_count > 0:
		summary["psnr_avg"] = psnr_total / float(psnr_count)
	return summary
