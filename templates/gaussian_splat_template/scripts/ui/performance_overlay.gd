extends MarginContainer
class_name GaussianPerformanceOverlay

@export var update_interval: float = 0.25
@export var show_vram_metrics: bool = true
@export var show_lod_metrics: bool = true
@export var show_streaming_metrics: bool = true
@export var show_compression_metrics: bool = false

const COMPUTE_POLICY_DEFAULT := 0
const COMPUTE_POLICY_FORCE_ON := 1
const COMPUTE_POLICY_FORCE_OFF := 2
const COMPUTE_POLICY_TOGGLE_KEY := KEY_F8

@onready var title_label: Label = get_node_or_null("Panel/VBox/TitleLabel")
@onready var body_label: RichTextLabel = get_node_or_null("Panel/VBox/Body")
@onready var footer_label: Label = get_node_or_null("Panel/VBox/Footer")
@export var camera_path: NodePath

var gaussian_node: GaussianSplatNode3D
var camera_node: Node3D
var _time_since_update := 0.0
var _ema_frame_ms := 0.0
var _ema_fps := 0.0
var _last_instant_ms := 0.0
var _last_instant_fps := 0.0
var _fps_samples: Array[float] = []
var _gpu_prefix_samples: Array[float] = []
const MAX_SAMPLES := 120

# Performance singleton reference
var _perf: Performance

## Color thresholds for metrics
const GPU_TIME_GOOD := 8.0      # <8ms = green (120+ FPS headroom)
const GPU_TIME_OK := 16.0       # 8-16ms = yellow (60-120 FPS)
const GPU_TIME_BAD := 33.0      # 16-33ms = orange (30-60 FPS)
# >33ms = red (<30 FPS)

const VRAM_USAGE_GOOD := 60.0   # <60% = white
const VRAM_USAGE_WARN := 80.0   # 60-80% = yellow
const VRAM_USAGE_CRITICAL := 95.0 # 80-95% = orange
# >95% = red

const BUFFER_USAGE_GOOD := 70.0  # <70% = white
const BUFFER_USAGE_WARN := 85.0  # 70-85% = yellow
# >85% = orange

## Enables processing so the overlay refreshes at runtime.
func _ready() -> void:
	_perf = Performance.get_singleton()
	set_process(true)
	set_process_unhandled_input(true)

## Assigns the Gaussian node used for statistics queries.
## @param node: GaussianSplatNode3D to monitor.
func set_gaussian_node(node: GaussianSplatNode3D) -> void:
	gaussian_node = node
	_try_resolve_camera()

## Assigns the camera rig node used for pose reporting.
## @param node: Camera rig node.
func set_camera_node(node: Node3D) -> void:
	camera_node = node

## Resolves the camera from the configured NodePath when missing.
func _try_resolve_camera() -> void:
	if camera_node:
		return
	if camera_path != NodePath():
		var node_obj = get_node_or_null(camera_path)
		if node_obj and node_obj is Node3D:
			camera_node = node_obj

## Tracks frame timing and refreshes the overlay at the configured interval.
## @param delta: Frame delta in seconds.
func _process(delta: float) -> void:
	_try_resolve_camera()
	if delta > 0.0:
		_last_instant_ms = delta * 1000.0
		_last_instant_fps = 1.0 / delta
		var alpha := 0.12
		_ema_frame_ms = (_ema_frame_ms == 0.0) ? _last_instant_ms : lerp(_ema_frame_ms, _last_instant_ms, alpha)
		_ema_fps = (_ema_fps == 0.0) ? _last_instant_fps : lerp(_ema_fps, _last_instant_fps, alpha)
		_fps_samples.append(_last_instant_fps)
		if _fps_samples.size() > MAX_SAMPLES:
			_fps_samples.pop_front()

	_time_since_update += delta
	if _time_since_update < update_interval:
		return
	_time_since_update = 0.0
	_refresh_overlay()

## Returns color-coded string for GPU timing (ms)
func _colorize_gpu_time(time_ms: float, value_str: String) -> String:
	if time_ms < GPU_TIME_GOOD:
		return "[color=green]%s[/color]" % value_str
	elif time_ms < GPU_TIME_OK:
		return "[color=yellow]%s[/color]" % value_str
	elif time_ms < GPU_TIME_BAD:
		return "[color=orange]%s[/color]" % value_str
	else:
		return "[color=red]%s[/color]" % value_str

## Returns color-coded string for VRAM usage percentage
func _colorize_vram_percent(percent: float, value_str: String) -> String:
	if percent < VRAM_USAGE_GOOD:
		return value_str
	elif percent < VRAM_USAGE_WARN:
		return "[color=yellow]%s[/color]" % value_str
	elif percent < VRAM_USAGE_CRITICAL:
		return "[color=orange]%s[/color]" % value_str
	else:
		return "[color=red]%s[/color]" % value_str

## Returns color-coded string for buffer usage percentage
func _colorize_buffer_percent(percent: float, value_str: String) -> String:
	if percent < BUFFER_USAGE_GOOD:
		return value_str
	elif percent < BUFFER_USAGE_WARN:
		return "[color=yellow]%s[/color]" % value_str
	else:
		return "[color=orange]%s[/color]" % value_str

## Returns color-coded string for LOD reduction percentage (higher = more aggressive = red)
func _colorize_lod_reduction(percent: float, value_str: String) -> String:
	if percent < 25.0:
		return "[color=green]%s[/color]" % value_str
	elif percent < 50.0:
		return "[color=yellow]%s[/color]" % value_str
	elif percent < 75.0:
		return "[color=orange]%s[/color]" % value_str
	else:
		return "[color=red]%s[/color]" % value_str

func _format_compute_policy(policy: int) -> String:
	match policy:
		COMPUTE_POLICY_FORCE_ON:
			return "force_on"
		COMPUTE_POLICY_FORCE_OFF:
			return "force_off"
		_:
			return "default"

func _toggle_compute_raster_policy() -> void:
	if not gaussian_node:
		return
	var renderer := gaussian_node.get_renderer()
	if renderer and renderer.has_method("set_debug_compute_raster_policy"):
		var current := COMPUTE_POLICY_DEFAULT
		if renderer.has_method("get_debug_compute_raster_policy"):
			current = int(renderer.get_debug_compute_raster_policy())
		var next := (current + 1) % 3
		renderer.set_debug_compute_raster_policy(next)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == COMPUTE_POLICY_TOGGLE_KEY:
			_toggle_compute_raster_policy()

## Rebuilds the overlay text with the latest renderer statistics using Custom Performance Monitors.
func _refresh_overlay() -> void:
	var lines: Array[String] = []

	# ========================================================================
	# FRAME & FPS
	# ========================================================================
	lines.append("[b]═══ FRAME ═══[/b]")
	lines.append("FPS: %.1f (inst) / %.1f (engine avg)" % [_last_instant_fps, Engine.get_frames_per_second()])
	lines.append("CPU frame: %.2f ms (inst) / %.2f ms (EMA)" % [_last_instant_ms, _ema_frame_ms])

	if _fps_samples.size() > 0:
		var fps_min = _fps_samples.min()
		var fps_max = _fps_samples.max()
		var fps_avg = 0.0
		for v in _fps_samples:
			fps_avg += v
		fps_avg /= _fps_samples.size()
		lines.append("FPS trends (%d frames): avg %.1f | min %.1f | max %.1f" % [_fps_samples.size(), fps_avg, fps_min, fps_max])

	# ========================================================================
	# CAMERA
	# ========================================================================
	if camera_node:
		var basis: Basis = camera_node.global_transform.basis
		var origin: Vector3 = camera_node.global_transform.origin
		var euler := basis.get_euler()
		lines.append("")
		lines.append("[b]═══ CAMERA ═══[/b]")
		lines.append("Pos: (%.2f, %.2f, %.2f)" % [origin.x, origin.y, origin.z])
		lines.append("Rot: (%.1f°, %.1f°, %.1f°)" % [rad_to_deg(euler.x), rad_to_deg(euler.y), rad_to_deg(euler.z)])
		var cam = camera_node.get_node_or_null("Camera3D")
		if cam and cam is Camera3D:
			var cam3d: Camera3D = cam
			var fov := cam3d.fov
			var size := cam3d.size
			var ortho := cam3d.projection == Camera3D.PROJECTION_ORTHOGONAL
			lines.append("Projection: %s | FOV: %.1f° | Size: %.2f" % ["Ortho" if ortho else "Persp", fov, size])

	# ========================================================================
	# GPU PIPELINE (Custom Performance Monitors)
	# ========================================================================
	lines.append("")
	lines.append("[b]═══ GPU PIPELINE ═══[/b]")

	var cpu_setup = _perf.get_custom_monitor("gaussian_splatting/cpu_setup_time_ms")
	var gpu_frustum = _perf.get_custom_monitor("gaussian_splatting/gpu_time_frustum_cull_ms")
	var gpu_binning = _perf.get_custom_monitor("gaussian_splatting/gpu_time_binning_ms")
	var gpu_prefix = _perf.get_custom_monitor("gaussian_splatting/gpu_time_prefix_ms")
	var gpu_sort = _perf.get_custom_monitor("gaussian_splatting/gpu_time_sort_ms")
	var gpu_raster = _perf.get_custom_monitor("gaussian_splatting/gpu_time_raster_ms")
	var gpu_total = _perf.get_custom_monitor("gaussian_splatting/gpu_time_frame_ms")

	lines.append("CPU setup: %s" % _colorize_gpu_time(cpu_setup, "%.3f ms" % cpu_setup))
	lines.append("GPU frustum cull: %s" % _colorize_gpu_time(gpu_frustum, "%.3f ms" % gpu_frustum))
	lines.append("GPU binning: %s" % _colorize_gpu_time(gpu_binning, "%.3f ms" % gpu_binning))
	lines.append("GPU prefix scan: %s" % _colorize_gpu_time(gpu_prefix, "%.3f ms" % gpu_prefix))
	lines.append("GPU sort: %s" % _colorize_gpu_time(gpu_sort, "%.3f ms" % gpu_sort))
	lines.append("GPU rasterize: %s" % _colorize_gpu_time(gpu_raster, "%.3f ms" % gpu_raster))
	lines.append("GPU total: %s" % _colorize_gpu_time(gpu_total, "%.3f ms" % gpu_total))

	# ========================================================================
	# PROJECTION & VISIBILITY
	# ========================================================================
	lines.append("")
	lines.append("[b]═══ PROJECTION & VISIBILITY ═══[/b]")

	var visible_splats = int(_perf.get_custom_monitor("gaussian_splatting/visible_splats"))
	var culled_frustum = int(_perf.get_custom_monitor("gaussian_splatting/culled_frustum"))
	var success_rate = _perf.get_custom_monitor("gaussian_splatting/projection_success_rate_pct")
	var near_clamp = int(_perf.get_custom_monitor("gaussian_splatting/projection_near_clamp_count"))
	var behind = int(_perf.get_custom_monitor("gaussian_splatting/projection_behind_camera_count"))
	var screen_cull = int(_perf.get_custom_monitor("gaussian_splatting/projection_screen_culled_count"))
	var extreme_aspect = int(_perf.get_custom_monitor("gaussian_splatting/projection_extreme_aspect_count"))

	lines.append("Visible splats: %s" % _format_number(visible_splats))
	lines.append("Frustum culled: %s" % _format_number(culled_frustum))

	var success_color = "green" if success_rate > 95.0 else ("yellow" if success_rate > 85.0 else "orange")
	lines.append("Projection success: [color=%s]%.1f%%[/color]" % [success_color, success_rate])
	lines.append("Rejects: near %s | behind %s | screen %s | aspect %s" % [
		_format_number(near_clamp),
		_format_number(behind),
		_format_number(screen_cull),
		_format_number(extreme_aspect)
	])

	var overlap_used = int(_perf.get_custom_monitor("gaussian_splatting/overlap_records_used"))
	var overlap_budget = int(_perf.get_custom_monitor("gaussian_splatting/overlap_record_budget"))
	var overlap_pct = (float(overlap_used) / float(overlap_budget) * 100.0) if overlap_budget > 0 else 0.0
	lines.append("Overlap: %s / %s (%s)" % [
		_format_number(overlap_used),
		_format_number(overlap_budget),
		_colorize_buffer_percent(overlap_pct, "%.1f%%" % overlap_pct)
	])

	# ========================================================================
	# VRAM BUDGET (if enabled)
	# ========================================================================
	if show_vram_metrics:
		lines.append("")
		lines.append("[b]═══ VRAM BUDGET ═══[/b]")

		var vram_usage = _perf.get_custom_monitor("gaussian_splatting/vram_current_usage_mb")
		var vram_budget = _perf.get_custom_monitor("gaussian_splatting/vram_budget_mb")
		var vram_percent = _perf.get_custom_monitor("gaussian_splatting/vram_usage_percent")
		var vram_warning = int(_perf.get_custom_monitor("gaussian_splatting/vram_budget_warning_active"))
		var vram_critical = int(_perf.get_custom_monitor("gaussian_splatting/vram_budget_critical_active"))

		var warning_icon = ""
		if vram_critical > 0:
			warning_icon = "[color=red]⚠ CRITICAL[/color] "
		elif vram_warning > 0:
			warning_icon = "[color=orange]⚠ WARNING[/color] "

		lines.append("%sUsage: %.1f / %.1f MB (%s)" % [
			warning_icon,
			vram_usage,
			vram_budget,
			_colorize_vram_percent(vram_percent, "%.1f%%" % vram_percent)
		])

		var vram_reserved = _perf.get_custom_monitor("gaussian_splatting/vram_reserved_for_streaming_mb")
		var vram_allocated = _perf.get_custom_monitor("gaussian_splatting/vram_allocated_chunks_mb")
		var vram_pool = _perf.get_custom_monitor("gaussian_splatting/vram_pool_size_mb")

		lines.append("Reserved: %.1f MB | Allocated: %.1f MB | Pool: %.1f MB" % [
			vram_reserved,
			vram_allocated,
			vram_pool
		])

		var thrashing = int(_perf.get_custom_monitor("gaussian_splatting/vram_thrashing_detected"))
		var evictions = int(_perf.get_custom_monitor("gaussian_splatting/vram_eviction_count"))

		if thrashing > 0:
			lines.append("[color=red]⚠ THRASHING DETECTED[/color] | Evictions: %d" % evictions)
		elif evictions > 0:
			lines.append("Evictions: %d" % evictions)

	# ========================================================================
	# LOD SYSTEM (if enabled)
	# ========================================================================
	if show_lod_metrics:
		lines.append("")
		lines.append("[b]═══ LOD SYSTEM ═══[/b]")

		var lod_level = int(_perf.get_custom_monitor("gaussian_splatting/lod_current_level"))
		var lod_reduction = _perf.get_custom_monitor("gaussian_splatting/lod_reduction_ratio_pct")
		var lod_min_dist = _perf.get_custom_monitor("gaussian_splatting/lod_min_chunk_distance")
		var lod_max_dist = _perf.get_custom_monitor("gaussian_splatting/lod_max_chunk_distance")
		var lod_avg_dist = _perf.get_custom_monitor("gaussian_splatting/lod_avg_chunk_distance")

		lines.append("LOD level (avg): %d | Reduction: %s" % [
			lod_level,
			_colorize_lod_reduction(lod_reduction, "%.1f%%" % lod_reduction)
		])

		if lod_min_dist > 0 or lod_max_dist > 0:
			lines.append("Distance: min %.1f | avg %.1f | max %.1f" % [lod_min_dist, lod_avg_dist, lod_max_dist])

		var skip_factor = int(_perf.get_custom_monitor("gaussian_splatting/lod_splat_skip_factor"))
		var opacity_mult = _perf.get_custom_monitor("gaussian_splatting/lod_opacity_multiplier")
		var chunks_transition = int(_perf.get_custom_monitor("gaussian_splatting/lod_chunks_in_transition"))

		lines.append("Splat skip: %dx | Opacity: %.2f | Transitioning: %d chunks" % [
			skip_factor,
			opacity_mult,
			chunks_transition
		])

		var quality_degrade = int(_perf.get_custom_monitor("gaussian_splatting/lod_quality_degradation_active"))
		if quality_degrade > 0:
			lines.append("[color=orange]⚠ Quality degradation active (VRAM pressure)[/color]")

	# ========================================================================
	# STREAMING (if enabled)
	# ========================================================================
	if show_streaming_metrics:
		lines.append("")
		lines.append("[b]═══ STREAMING ═══[/b]")

		var visible_chunks = int(_perf.get_custom_monitor("gaussian_splatting/streaming_visible_chunks"))
		var loaded_chunks = int(_perf.get_custom_monitor("gaussian_splatting/streaming_loaded_chunks"))
		var total_chunks = int(_perf.get_custom_monitor("gaussian_splatting/streaming_total_chunks"))
		var pending_loads = int(_perf.get_custom_monitor("gaussian_splatting/streaming_pending_chunk_loads"))

		lines.append("Chunks: visible %d | loaded %d / %d" % [visible_chunks, loaded_chunks, total_chunks])

		if pending_loads > 0:
			lines.append("Pending loads: [color=yellow]%d[/color]" % pending_loads)

		var buffer_slots = int(_perf.get_custom_monitor("gaussian_splatting/streaming_buffer_slot_count"))
		var buffer_used = int(_perf.get_custom_monitor("gaussian_splatting/streaming_buffer_slots_used"))
		var buffer_pct = (float(buffer_used) / float(buffer_slots) * 100.0) if buffer_slots > 0 else 0.0

		lines.append("Buffer slots: %d / %d (%s)" % [
			buffer_used,
			buffer_slots,
			_colorize_buffer_percent(buffer_pct, "%.1f%%" % buffer_pct)
		])

		# Memory Stream stats
		var upload_mb = _perf.get_custom_monitor("gaussian_splatting/memory_stream_total_bytes_uploaded_mb")
		var upload_rate = _perf.get_custom_monitor("gaussian_splatting/memory_stream_upload_rate_mbps")
		var stall_pct = _perf.get_custom_monitor("gaussian_splatting/memory_stream_stall_percent")
		var effective_upload_cap_frame = _perf.get_custom_monitor("gaussian_splatting/streaming_effective_upload_cap_mb_per_frame")
		var effective_upload_cap_slice = _perf.get_custom_monitor("gaussian_splatting/streaming_effective_upload_cap_mb_per_slice")
		var effective_upload_cap_bandwidth = _perf.get_custom_monitor("gaussian_splatting/streaming_effective_upload_cap_mb_per_second")
		var effective_vram_budget_mb = _perf.get_custom_monitor("gaussian_splatting/streaming_effective_vram_budget_mb")
		var effective_vram_max_chunks = int(_perf.get_custom_monitor("gaussian_splatting/streaming_effective_vram_max_chunks"))
		var upload_frame_cap_hit = int(_perf.get_custom_monitor("gaussian_splatting/streaming_upload_frame_cap_hit"))
		var upload_bandwidth_cap_hit = int(_perf.get_custom_monitor("gaussian_splatting/streaming_upload_bandwidth_cap_hit"))
		var chunk_load_cap_hit = int(_perf.get_custom_monitor("gaussian_splatting/streaming_chunk_load_cap_hit"))
		var vram_chunk_cap_hit = int(_perf.get_custom_monitor("gaussian_splatting/streaming_vram_chunk_cap_hit"))
		var queue_pressure_active = int(_perf.get_custom_monitor("gaussian_splatting/streaming_queue_pressure_active"))

		lines.append("Upload: %.2f MB total | %.2f MB/s" % [upload_mb, upload_rate])
		lines.append("Caps: frame %.0f MB | slice %.0f MB | bandwidth %.0f MB/s" % [
			effective_upload_cap_frame,
			effective_upload_cap_slice,
			effective_upload_cap_bandwidth
		])
		lines.append("VRAM cap: %.0f MB | max chunks %d" % [effective_vram_budget_mb, effective_vram_max_chunks])

		var cap_markers := PackedStringArray()
		if upload_frame_cap_hit > 0:
			cap_markers.append("upload/frame")
		if upload_bandwidth_cap_hit > 0:
			cap_markers.append("upload/bandwidth")
		if chunk_load_cap_hit > 0:
			cap_markers.append("chunk-load")
		if vram_chunk_cap_hit > 0:
			cap_markers.append("vram")
		if queue_pressure_active > 0:
			cap_markers.append("queue")
		if cap_markers.size() > 0:
			lines.append("[color=orange]Pressure: %s[/color]" % ", ".join(cap_markers))

		if stall_pct > 5.0:
			lines.append("Pipeline stalls: [color=orange]%.1f%%[/color]" % stall_pct)
		elif stall_pct > 0.1:
			lines.append("Pipeline stalls: %.1f%%" % stall_pct)

	# ========================================================================
	# COMPRESSION (if enabled and active)
	# ========================================================================
	if show_compression_metrics:
		var sh_raw = _perf.get_custom_monitor("gaussian_splatting/sh_compression_raw_mb")
		var sh_compressed = _perf.get_custom_monitor("gaussian_splatting/sh_compression_compressed_mb")
		var sh_ratio = _perf.get_custom_monitor("gaussian_splatting/sh_compression_ratio_pct")

		if sh_raw > 0:
			lines.append("")
			lines.append("[b]═══ SH COMPRESSION ═══[/b]")
			var savings_mb = sh_raw - sh_compressed
			var savings_color = "green" if sh_ratio > 50.0 else "yellow"
			lines.append("%.2f MB → %.2f MB ([color=%s]%.1f%% compressed[/color])" % [
				sh_raw,
				sh_compressed,
				savings_color,
				sh_ratio
			])
			lines.append("VRAM saved: [color=green]%.2f MB[/color]" % savings_mb)

	# ========================================================================
	# NODE STATISTICS (legacy for compatibility)
	# ========================================================================
	if gaussian_node:
		var node_stats := gaussian_node.get_statistics()
		lines.append("")
		lines.append("[b]═══ NODE ═══[/b]")
		lines.append("Total splats: %s" % _format_number(node_stats.get("total_splats", 0)))
		lines.append("Last update: %.2f ms" % node_stats.get("update_time_ms", 0.0))

		var renderer := gaussian_node.get_renderer()
		if renderer:
			if renderer.has_method("get_debug_compute_raster_policy"):
				var policy := int(renderer.get_debug_compute_raster_policy())
				lines.append("Raster policy: %s (F8 to toggle)" % _format_compute_policy(policy))

	# ========================================================================
	# GLOBAL STATISTICS
	# ========================================================================
	var bootstrap = get_tree().get_root().get_node_or_null("GaussianBootstrap")
	if bootstrap and bootstrap.is_ready:
		var global_stats: Dictionary = bootstrap.get_global_stats()
		if not global_stats.is_empty():
			lines.append("")
			lines.append("[b]═══ GLOBAL ═══[/b]")
			lines.append("Active gaussians: %s" % _format_number(global_stats.get('total_gaussians', 0)))
			lines.append("GPU memory: %.2f MB" % global_stats.get('total_memory_mb', 0.0))
			lines.append("Registered buffers: %s" % global_stats.get('buffer_count', 0))
			lines.append("GPU sorting: %s" % ("enabled" if global_stats.get('gpu_sorting_enabled', false) else "disabled"))

	if body_label:
		body_label.text = "\n".join(lines)

	if footer_label:
		footer_label.text = "WASD: move | Space/C: ascend/descend | Shift: boost | RMB: orbit | MMB: pan | Wheel: zoom | F8: raster policy"

## Formats large numbers with K/M suffixes for readability
func _format_number(value: int) -> String:
	if value >= 1000000:
		return "%.2fM" % (value / 1000000.0)
	elif value >= 1000:
		return "%.1fK" % (value / 1000.0)
	else:
		return str(value)
