extends MarginContainer
class_name GaussianPerformanceOverlay

@export var update_interval: float = 0.1
@export var show_scene_name: bool = true

var _time_since_update := 0.0
var _ema_fps := 0.0
var _scene_label: Label = null

const GREEN := Color(0.29, 0.87, 0.50)
const YELLOW := Color(0.98, 0.75, 0.14)
const RED := Color(0.97, 0.44, 0.44)
const WHITE := Color(0.85, 0.87, 0.9)
const MUTED := Color(0.55, 0.58, 0.62)

const VRAM_FIELDS := [
	"vram_current_usage_mb",
	"vram_budget_mb",
	"vram_usage_percent",
	"vram_current_max_chunks",
	"vram_loaded_chunks",
	"vram_evicted_this_frame",
	"vram_loaded_this_frame",
	"vram_budget_warning_active",
	"vram_regulation_adjustments",
	"vram_thrashing_events",
]

const MEMORY_STREAM_FIELDS := [
	"memory_stream_total_bytes_uploaded_mb",
	"memory_stream_total_bytes_downloaded_mb",
	"memory_stream_buffer_switches",
	"memory_stream_stalls",
	"memory_stream_stall_percent",
	"memory_stream_pool_hits",
	"memory_stream_pool_misses",
	"memory_stream_pool_hit_rate_pct",
	"memory_stream_peak_memory_mb",
	"memory_stream_defrag_count",
]

const STREAMING_FIELDS := [
	"streaming_total_chunks",
	"streaming_visible_chunks",
	"streaming_loaded_chunks",
	"streaming_frustum_culled_chunks",
	"streaming_vram_usage_mb",
	"streaming_chunks_loaded_this_frame",
	"streaming_chunks_evicted_this_frame",
	"streaming_visible_count",
	"streaming_buffer_capacity_splats",
	"streaming_effective_splat_count",
	"streaming_visible_change_ratio",
	"streaming_lod_blend_factor",
	"streaming_sh_band_level",
	"streaming_bytes_uploaded_mb",
	"streaming_buffer_switches",
]

const LOD_FIELDS := [
	"lod_current_level",
	"lod_distance_multiplier",
	"lod_target_distance",
	"lod_hysteresis_zone",
	"lod_blend_distance",
	"lod_transitions_this_frame",
	"lod_splat_skip_factor",
	"lod_opacity_multiplier",
	"lod_effective_count_after_skip",
	"lod_chunk_blend_factors_avg",
	"lod_chunks_in_transition",
	"lod_quality_degradation_active",
	"lod_min_chunk_distance",
	"lod_max_chunk_distance",
	"lod_avg_chunk_distance",
	"lod_reduction_ratio_pct",
	"lod_level_0_chunk_count",
	"lod_sh_band_3_chunk_count",
]

const PREFETCH_FIELDS := [
	"chunk_prefetch_hits",
	"chunk_prefetch_misses",
	"chunk_prefetch_efficiency_pct",
	"chunk_camera_velocity",
	"chunk_average_load_time_ms",
	"chunk_upload_queue_depth",
	"chunk_pack_jobs_in_flight",
	"chunk_total_capacity_mb",
]

const PACK_FIELDS := [
	"pack_avg_time_ms",
	"pack_max_time_ms",
	"pack_jobs_completed",
	"upload_mb_this_frame",
	"upload_chunks_this_frame",
]

const SH_COMPRESSION_FIELDS := [
	"sh_compression_raw_mb",
	"sh_compression_compressed_mb",
	"sh_compression_ratio_pct",
]

# Node path mapping for the new 2-column layout
var _node_paths := {
	# Left column - GPU
	"gpu_time_frame_ms": "Panel/Columns/Left/GPU/gpu_time_frame_ms",
	"gpu_time_cull_ms": "Panel/Columns/Left/GPU/gpu_time_cull_ms",
	"gpu_time_binning_ms": "Panel/Columns/Left/GPU/gpu_time_binning_ms",
	"gpu_time_prefix_ms": "Panel/Columns/Left/GPU/gpu_time_prefix_ms",
	"gpu_time_raster_ms": "Panel/Columns/Left/GPU/gpu_time_raster_ms",
	"gpu_time_resolve_ms": "Panel/Columns/Left/GPU/gpu_time_resolve_ms",
	"cpu_setup_time_ms": "Panel/Columns/Left/GPU/cpu_setup_time_ms",
	# Left column - Visibility
	"visible_splats": "Panel/Columns/Left/VIS/visible_splats",
	"total_processed": "Panel/Columns/Left/VIS/total_processed",
	"projection_success_count": "Panel/Columns/Left/VIS/projection_success_count",
	"projection_success_rate_pct": "Panel/Columns/Left/VIS/projection_success_rate_pct",
	# Left column - Rejections
	"clip_reject_count": "Panel/Columns/Left/REJ/clip_reject_count",
	"radius_reject_count": "Panel/Columns/Left/REJ/radius_reject_count",
	"viewport_reject_count": "Panel/Columns/Left/REJ/viewport_reject_count",
	"extreme_aspect_count": "Panel/Columns/Left/REJ/extreme_aspect_count",
	"index_mismatch_count": "Panel/Columns/Left/REJ/index_mismatch_count",
	# Left column - Tiles
	"tile_count": "Panel/Columns/Left/TILE/tile_count",
	"overflow_tile_count": "Panel/Columns/Left/TILE/overflow_tile_count",
	"clamped_records": "Panel/Columns/Left/TILE/clamped_records",
	"aggregated_count": "Panel/Columns/Left/TILE/aggregated_count",
	# Left column - VRAM
	"vram_current_usage_mb": "Panel/Columns/Left/VRAM/vram_current_usage_mb",
	"vram_budget_mb": "Panel/Columns/Left/VRAM/vram_budget_mb",
	"vram_usage_percent": "Panel/Columns/Left/VRAM/vram_usage_percent",
	"vram_current_max_chunks": "Panel/Columns/Left/VRAM/vram_current_max_chunks",
	"vram_loaded_chunks": "Panel/Columns/Left/VRAM/vram_loaded_chunks",
	"vram_evicted_this_frame": "Panel/Columns/Left/VRAM/vram_evicted_this_frame",
	"vram_loaded_this_frame": "Panel/Columns/Left/VRAM/vram_loaded_this_frame",
	"vram_budget_warning_active": "Panel/Columns/Left/VRAM/vram_budget_warning_active",
	"vram_regulation_adjustments": "Panel/Columns/Left/VRAM/vram_regulation_adjustments",
	"vram_thrashing_events": "Panel/Columns/Left/VRAM/vram_thrashing_events",
	# Left column - Memory Stream
	"memory_stream_total_bytes_uploaded_mb": "Panel/Columns/Left/MEM/memory_stream_total_bytes_uploaded_mb",
	"memory_stream_total_bytes_downloaded_mb": "Panel/Columns/Left/MEM/memory_stream_total_bytes_downloaded_mb",
	"memory_stream_buffer_switches": "Panel/Columns/Left/MEM/memory_stream_buffer_switches",
	"memory_stream_stalls": "Panel/Columns/Left/MEM/memory_stream_stalls",
	"memory_stream_stall_percent": "Panel/Columns/Left/MEM/memory_stream_stall_percent",
	"memory_stream_pool_hits": "Panel/Columns/Left/MEM/memory_stream_pool_hits",
	"memory_stream_pool_misses": "Panel/Columns/Left/MEM/memory_stream_pool_misses",
	"memory_stream_pool_hit_rate_pct": "Panel/Columns/Left/MEM/memory_stream_pool_hit_rate_pct",
	"memory_stream_peak_memory_mb": "Panel/Columns/Left/MEM/memory_stream_peak_memory_mb",
	"memory_stream_defrag_count": "Panel/Columns/Left/MEM/memory_stream_defrag_count",
	# Right column - Streaming
	"streaming_total_chunks": "Panel/Columns/Right/STREAM/streaming_total_chunks",
	"streaming_visible_chunks": "Panel/Columns/Right/STREAM/streaming_visible_chunks",
	"streaming_loaded_chunks": "Panel/Columns/Right/STREAM/streaming_loaded_chunks",
	"streaming_frustum_culled_chunks": "Panel/Columns/Right/STREAM/streaming_frustum_culled_chunks",
	"streaming_vram_usage_mb": "Panel/Columns/Right/STREAM/streaming_vram_usage_mb",
	"streaming_chunks_loaded_this_frame": "Panel/Columns/Right/STREAM/streaming_chunks_loaded_this_frame",
	"streaming_chunks_evicted_this_frame": "Panel/Columns/Right/STREAM/streaming_chunks_evicted_this_frame",
	"streaming_visible_count": "Panel/Columns/Right/STREAM/streaming_visible_count",
	"streaming_buffer_capacity_splats": "Panel/Columns/Right/STREAM/streaming_buffer_capacity_splats",
	"streaming_effective_splat_count": "Panel/Columns/Right/STREAM/streaming_effective_splat_count",
	"streaming_visible_change_ratio": "Panel/Columns/Right/STREAM/streaming_visible_change_ratio",
	"streaming_lod_blend_factor": "Panel/Columns/Right/STREAM/streaming_lod_blend_factor",
	"streaming_sh_band_level": "Panel/Columns/Right/STREAM/streaming_sh_band_level",
	"streaming_bytes_uploaded_mb": "Panel/Columns/Right/STREAM/streaming_bytes_uploaded_mb",
	"streaming_buffer_switches": "Panel/Columns/Right/STREAM/streaming_buffer_switches",
	# Right column - LOD
	"lod_current_level": "Panel/Columns/Right/LOD/lod_current_level",
	"lod_distance_multiplier": "Panel/Columns/Right/LOD/lod_distance_multiplier",
	"lod_target_distance": "Panel/Columns/Right/LOD/lod_target_distance",
	"lod_hysteresis_zone": "Panel/Columns/Right/LOD/lod_hysteresis_zone",
	"lod_blend_distance": "Panel/Columns/Right/LOD/lod_blend_distance",
	"lod_transitions_this_frame": "Panel/Columns/Right/LOD/lod_transitions_this_frame",
	"lod_splat_skip_factor": "Panel/Columns/Right/LOD/lod_splat_skip_factor",
	"lod_opacity_multiplier": "Panel/Columns/Right/LOD/lod_opacity_multiplier",
	"lod_effective_count_after_skip": "Panel/Columns/Right/LOD/lod_effective_count_after_skip",
	"lod_chunk_blend_factors_avg": "Panel/Columns/Right/LOD/lod_chunk_blend_factors_avg",
	"lod_chunks_in_transition": "Panel/Columns/Right/LOD/lod_chunks_in_transition",
	"lod_quality_degradation_active": "Panel/Columns/Right/LOD/lod_quality_degradation_active",
	"lod_min_chunk_distance": "Panel/Columns/Right/LOD/lod_min_chunk_distance",
	"lod_max_chunk_distance": "Panel/Columns/Right/LOD/lod_max_chunk_distance",
	"lod_avg_chunk_distance": "Panel/Columns/Right/LOD/lod_avg_chunk_distance",
	"lod_reduction_ratio_pct": "Panel/Columns/Right/LOD/lod_reduction_ratio_pct",
	"lod_level_0_chunk_count": "Panel/Columns/Right/LOD/lod_level_0_chunk_count",
	"lod_sh_band_3_chunk_count": "Panel/Columns/Right/LOD/lod_sh_band_3_chunk_count",
	# Right column - Prefetch
	"chunk_prefetch_hits": "Panel/Columns/Right/PREFETCH/chunk_prefetch_hits",
	"chunk_prefetch_misses": "Panel/Columns/Right/PREFETCH/chunk_prefetch_misses",
	"chunk_prefetch_efficiency_pct": "Panel/Columns/Right/PREFETCH/chunk_prefetch_efficiency_pct",
	"chunk_camera_velocity": "Panel/Columns/Right/PREFETCH/chunk_camera_velocity",
	"chunk_average_load_time_ms": "Panel/Columns/Right/PREFETCH/chunk_average_load_time_ms",
	"chunk_upload_queue_depth": "Panel/Columns/Right/PREFETCH/chunk_upload_queue_depth",
	"chunk_pack_jobs_in_flight": "Panel/Columns/Right/PREFETCH/chunk_pack_jobs_in_flight",
	"chunk_total_capacity_mb": "Panel/Columns/Right/PREFETCH/chunk_total_capacity_mb",
	# Right column - Pack Timing
	"pack_avg_time_ms": "Panel/Columns/Right/PACK/pack_avg_time_ms",
	"pack_max_time_ms": "Panel/Columns/Right/PACK/pack_max_time_ms",
	"pack_jobs_completed": "Panel/Columns/Right/PACK/pack_jobs_completed",
	"upload_mb_this_frame": "Panel/Columns/Right/PACK/upload_mb_this_frame",
	"upload_chunks_this_frame": "Panel/Columns/Right/PACK/upload_chunks_this_frame",
	# Right column - SH Cache
	"sh_cache_hits": "Panel/Columns/Right/SH/sh_cache_hits",
	"sh_cache_updates": "Panel/Columns/Right/SH/sh_cache_updates",
	"sh_cache_hit_rate_pct": "Panel/Columns/Right/SH/sh_cache_hit_rate_pct",
	"sh_compression_raw_mb": "Panel/Columns/Right/SH/sh_compression_raw_mb",
	"sh_compression_compressed_mb": "Panel/Columns/Right/SH/sh_compression_compressed_mb",
	"sh_compression_ratio_pct": "Panel/Columns/Right/SH/sh_compression_ratio_pct",
}

func _ready() -> void:
	set_process(true)
	_setup_scene_label()

func _setup_scene_label() -> void:
	if not show_scene_name:
		return
	# Create scene name label above FPS
	var fps_node = get_node_or_null("Panel/Columns/Left/FPS")
	if fps_node and fps_node.get_parent():
		_scene_label = Label.new()
		_scene_label.name = "SceneName"
		_scene_label.add_theme_font_size_override("font_size", 10)
		_scene_label.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4, 1.0))
		_scene_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		_scene_label.text = _get_current_scene_name()
		fps_node.get_parent().add_child(_scene_label)
		fps_node.get_parent().move_child(_scene_label, 0)

func _get_current_scene_name() -> String:
	var tree = get_tree()
	if tree and tree.current_scene:
		var scene_path = tree.current_scene.scene_file_path
		if scene_path.is_empty():
			return tree.current_scene.name
		return scene_path.get_file().get_basename()
	return "Unknown"

func _process(delta: float) -> void:
	var engine_fps := float(Engine.get_frames_per_second())
	if engine_fps > 0.0:
		_ema_fps = engine_fps
	elif delta > 0.0:
		_ema_fps = 1.0 / delta
	_time_since_update += delta
	if _time_since_update < update_interval:
		return
	_time_since_update = 0.0
	_refresh()

func _m(name: String) -> float:
	if Performance.has_custom_monitor("gaussian_splatting/" + name):
		return Performance.get_custom_monitor("gaussian_splatting/" + name)
	return 0.0

func _has_monitor(name: String) -> bool:
	return Performance.has_custom_monitor("gaussian_splatting/" + name)

func _n(val: float) -> String:
	if val >= 1000000: return "%.2fM" % (val / 1000000.0)
	if val >= 1000: return "%.1fK" % (val / 1000.0)
	return str(int(val))

func _set_label(name: String, text: String, color: Color = WHITE) -> void:
	if _node_paths.has(name):
		var node = get_node_or_null(_node_paths[name])
		if node:
			node.text = text
			node.add_theme_color_override("font_color", color)

func _set_group_na(names: Array) -> void:
	for name in names:
		_set_label(name, "N/A", MUTED)

func _time_color(ms: float) -> Color:
	if ms < 8.0: return GREEN
	if ms < 16.0: return YELLOW
	return RED

func _pct_color(pct: float, good_above: float = 90.0) -> Color:
	if pct >= good_above: return GREEN
	if pct >= good_above * 0.7: return YELLOW
	return RED

func _is_streaming_active() -> bool:
	if _has_monitor("streaming_monitor_ready") and int(_m("streaming_monitor_ready")) == 1:
		return true
	return _m("streaming_total_chunks") > 0.0 \
		or _m("streaming_loaded_chunks") > 0.0 \
		or _m("streaming_visible_chunks") > 0.0

func _refresh() -> void:
	# Scene Name
	if _scene_label and show_scene_name:
		_scene_label.text = _get_current_scene_name()

	# FPS
	var fps_color = GREEN if _ema_fps >= 55 else (YELLOW if _ema_fps >= 30 else RED)
	var fps_node = get_node_or_null("Panel/Columns/Left/FPS")
	if fps_node:
		fps_node.text = "%.0f FPS" % _ema_fps
		fps_node.add_theme_color_override("font_color", fps_color)

	# GPU Timing
	var frame = _m("gpu_time_frame_ms")
	var cull = _m("gpu_time_cull_ms")
	var bin = _m("gpu_time_binning_ms")
	var prefix = _m("gpu_time_prefix_ms")
	var raster = _m("gpu_time_raster_ms")
	var resolve = _m("gpu_time_resolve_ms")
	var cpu_setup = _m("cpu_setup_time_ms")

	_set_label("gpu_time_frame_ms", "%.2f ms" % frame, _time_color(frame))
	_set_label("gpu_time_cull_ms", "%.2f ms" % cull, _time_color(cull))
	_set_label("gpu_time_binning_ms", "%.2f ms" % bin, _time_color(bin))
	_set_label("gpu_time_prefix_ms", "%.2f ms" % prefix, _time_color(prefix))
	_set_label("gpu_time_raster_ms", "%.2f ms" % raster, _time_color(raster))
	_set_label("gpu_time_resolve_ms", "%.2f ms" % resolve, _time_color(resolve))
	_set_label("cpu_setup_time_ms", "%.2f ms" % cpu_setup, _time_color(cpu_setup))

	# Visibility
	_set_label("visible_splats", _n(_m("visible_splats")))
	_set_label("total_processed", _n(_m("total_processed")))
	_set_label("projection_success_count", _n(_m("projection_success_count")))
	var success_pct = _m("projection_success_rate_pct")
	_set_label("projection_success_rate_pct", "%.1f%%" % success_pct, _pct_color(success_pct))

	# Rejections
	_set_label("clip_reject_count", _n(_m("clip_reject_count")))
	_set_label("radius_reject_count", _n(_m("radius_reject_count")))
	_set_label("viewport_reject_count", _n(_m("viewport_reject_count")))
	_set_label("extreme_aspect_count", _n(_m("extreme_aspect_count")))
	_set_label("index_mismatch_count", _n(_m("index_mismatch_count")))

	# Tiles
	_set_label("tile_count", _n(_m("tile_count")))
	_set_label("overflow_tile_count", _n(_m("overflow_tile_count")))
	_set_label("clamped_records", _n(_m("clamped_records")))
	_set_label("aggregated_count", _n(_m("aggregated_count")))

	var streaming_active = _is_streaming_active()

	# VRAM
	if streaming_active:
		_set_label("vram_current_usage_mb", "%.1f MB" % _m("vram_current_usage_mb"))
		_set_label("vram_budget_mb", "%.1f MB" % _m("vram_budget_mb"))
		var vram_pct = _m("vram_usage_percent")
		_set_label("vram_usage_percent", "%.1f%%" % vram_pct, _pct_color(100.0 - vram_pct, 50.0))
		_set_label("vram_current_max_chunks", _n(_m("vram_current_max_chunks")))
		_set_label("vram_loaded_chunks", _n(_m("vram_loaded_chunks")))
		_set_label("vram_evicted_this_frame", _n(_m("vram_evicted_this_frame")))
		_set_label("vram_loaded_this_frame", _n(_m("vram_loaded_this_frame")))
		_set_label("vram_budget_warning_active", "%d" % int(_m("vram_budget_warning_active")))
		_set_label("vram_regulation_adjustments", _n(_m("vram_regulation_adjustments")))
		_set_label("vram_thrashing_events", _n(_m("vram_thrashing_events")))
	else:
		_set_group_na(VRAM_FIELDS)

	# Memory Stream
	if streaming_active:
		_set_label("memory_stream_total_bytes_uploaded_mb", "%.2f MB" % _m("memory_stream_total_bytes_uploaded_mb"))
		_set_label("memory_stream_total_bytes_downloaded_mb", "%.2f MB" % _m("memory_stream_total_bytes_downloaded_mb"))
		_set_label("memory_stream_buffer_switches", _n(_m("memory_stream_buffer_switches")))
		_set_label("memory_stream_stalls", _n(_m("memory_stream_stalls")))
		var stall_pct = _m("memory_stream_stall_percent")
		_set_label("memory_stream_stall_percent", "%.1f%%" % stall_pct, _pct_color(100.0 - stall_pct, 95.0))
		_set_label("memory_stream_pool_hits", _n(_m("memory_stream_pool_hits")))
		_set_label("memory_stream_pool_misses", _n(_m("memory_stream_pool_misses")))
		var pool_hit = _m("memory_stream_pool_hit_rate_pct")
		_set_label("memory_stream_pool_hit_rate_pct", "%.1f%%" % pool_hit, _pct_color(pool_hit))
		_set_label("memory_stream_peak_memory_mb", "%.1f MB" % _m("memory_stream_peak_memory_mb"))
		_set_label("memory_stream_defrag_count", _n(_m("memory_stream_defrag_count")))
	else:
		_set_group_na(MEMORY_STREAM_FIELDS)

	# Streaming
	if streaming_active:
		_set_label("streaming_total_chunks", _n(_m("streaming_total_chunks")))
		_set_label("streaming_visible_chunks", _n(_m("streaming_visible_chunks")))
		_set_label("streaming_loaded_chunks", _n(_m("streaming_loaded_chunks")))
		_set_label("streaming_frustum_culled_chunks", _n(_m("streaming_frustum_culled_chunks")))
		_set_label("streaming_vram_usage_mb", "%.1f MB" % _m("streaming_vram_usage_mb"))
		_set_label("streaming_chunks_loaded_this_frame", _n(_m("streaming_chunks_loaded_this_frame")))
		_set_label("streaming_chunks_evicted_this_frame", _n(_m("streaming_chunks_evicted_this_frame")))
		_set_label("streaming_visible_count", _n(_m("streaming_visible_count")))
		_set_label("streaming_buffer_capacity_splats", _n(_m("streaming_buffer_capacity_splats")))
		_set_label("streaming_effective_splat_count", _n(_m("streaming_effective_splat_count")))
		_set_label("streaming_visible_change_ratio", "%.2f" % _m("streaming_visible_change_ratio"))
		_set_label("streaming_lod_blend_factor", "%.2f" % _m("streaming_lod_blend_factor"))
		_set_label("streaming_sh_band_level", "%d" % int(_m("streaming_sh_band_level")))
		_set_label("streaming_bytes_uploaded_mb", "%.2f MB" % _m("streaming_bytes_uploaded_mb"))
		_set_label("streaming_buffer_switches", _n(_m("streaming_buffer_switches")))
	else:
		_set_group_na(STREAMING_FIELDS)

	# LOD
	if streaming_active:
		_set_label("lod_current_level", "%d" % int(_m("lod_current_level")))
		_set_label("lod_distance_multiplier", "%.2f" % _m("lod_distance_multiplier"))
		_set_label("lod_target_distance", "%.1f" % _m("lod_target_distance"))
		_set_label("lod_hysteresis_zone", "%.2f" % _m("lod_hysteresis_zone"))
		_set_label("lod_blend_distance", "%.1f" % _m("lod_blend_distance"))
		_set_label("lod_transitions_this_frame", "%d" % int(_m("lod_transitions_this_frame")))
		_set_label("lod_splat_skip_factor", "%.2f" % _m("lod_splat_skip_factor"))
		_set_label("lod_opacity_multiplier", "%.2f" % _m("lod_opacity_multiplier"))
		_set_label("lod_effective_count_after_skip", _n(_m("lod_effective_count_after_skip")))
		_set_label("lod_chunk_blend_factors_avg", "%.2f" % _m("lod_chunk_blend_factors_avg"))
		_set_label("lod_chunks_in_transition", "%d" % int(_m("lod_chunks_in_transition")))
		_set_label("lod_quality_degradation_active", "%d" % int(_m("lod_quality_degradation_active")))
		_set_label("lod_min_chunk_distance", "%.1f" % _m("lod_min_chunk_distance"))
		_set_label("lod_max_chunk_distance", "%.1f" % _m("lod_max_chunk_distance"))
		_set_label("lod_avg_chunk_distance", "%.1f" % _m("lod_avg_chunk_distance"))
		var reduction = _m("lod_reduction_ratio_pct")
		_set_label("lod_reduction_ratio_pct", "%.1f%%" % reduction)
		_set_label("lod_level_0_chunk_count", "%d" % int(_m("lod_level_0_chunk_count")))
		_set_label("lod_sh_band_3_chunk_count", "%d" % int(_m("lod_sh_band_3_chunk_count")))
	else:
		_set_group_na(LOD_FIELDS)

	# Prefetch
	if streaming_active:
		_set_label("chunk_prefetch_hits", _n(_m("chunk_prefetch_hits")))
		_set_label("chunk_prefetch_misses", _n(_m("chunk_prefetch_misses")))
		var prefetch_eff = _m("chunk_prefetch_efficiency_pct")
		_set_label("chunk_prefetch_efficiency_pct", "%.1f%%" % prefetch_eff, _pct_color(prefetch_eff))
		_set_label("chunk_camera_velocity", "%.2f" % _m("chunk_camera_velocity"))
		_set_label("chunk_average_load_time_ms", "%.2f ms" % _m("chunk_average_load_time_ms"))
		_set_label("chunk_upload_queue_depth", "%d" % int(_m("chunk_upload_queue_depth")))
		_set_label("chunk_pack_jobs_in_flight", "%d" % int(_m("chunk_pack_jobs_in_flight")))
		_set_label("chunk_total_capacity_mb", "%.1f MB" % _m("chunk_total_capacity_mb"))
	else:
		_set_group_na(PREFETCH_FIELDS)

	# Pack Timing
	if streaming_active:
		var pack_avg = _m("pack_avg_time_ms")
		_set_label("pack_avg_time_ms", "%.2f ms" % pack_avg, _time_color(pack_avg))
		var pack_max = _m("pack_max_time_ms")
		_set_label("pack_max_time_ms", "%.2f ms" % pack_max, _time_color(pack_max))
		_set_label("pack_jobs_completed", "%d" % int(_m("pack_jobs_completed")))
		_set_label("upload_mb_this_frame", "%.2f MB" % _m("upload_mb_this_frame"))
		_set_label("upload_chunks_this_frame", "%d" % int(_m("upload_chunks_this_frame")))
	else:
		_set_group_na(PACK_FIELDS)

	# SH Cache
	_set_label("sh_cache_hits", _n(_m("sh_cache_hits")))
	_set_label("sh_cache_updates", _n(_m("sh_cache_updates")))
	var sh_hit = _m("sh_cache_hit_rate_pct")
	_set_label("sh_cache_hit_rate_pct", "%.1f%%" % sh_hit, _pct_color(sh_hit))
	if streaming_active:
		_set_label("sh_compression_raw_mb", "%.2f MB" % _m("sh_compression_raw_mb"))
		_set_label("sh_compression_compressed_mb", "%.2f MB" % _m("sh_compression_compressed_mb"))
		var sh_ratio = _m("sh_compression_ratio_pct")
		_set_label("sh_compression_ratio_pct", "%.1f%%" % sh_ratio, _pct_color(sh_ratio, 50.0))
	else:
		_set_group_na(SH_COMPRESSION_FIELDS)
