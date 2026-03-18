extends "res://scripts/qa_test_base.gd"
## Streaming Visual Smoke Test: Confirms streamed splats are visibly rendered.

@export var move_delay_frames: int = 10
@export var capture_delay_frames: int = 20
@export var capture_delay_seconds: float = 1.0
@export var capture_timeout_seconds: float = 8.0
@export var far_position: Vector3 = Vector3(0.0, 5.0, 80.0)
@export var near_position: Vector3 = Vector3(0.0, 5.0, 25.0)
@export var patch_half_extent: int = 12
@export var min_center_luma_variance: float = 0.0005
@export var min_center_to_corner_delta: float = 0.04
@export var min_global_luma_variance: float = 0.0002
@export var min_capture_attempts: int = 2

var world_node: GaussianSplatWorld3D
var camera: Camera3D
var _renderer: Object
var _focus_point := Vector3.ZERO
var _far_camera_world_position := Vector3.ZERO
var _near_camera_world_position := Vector3.ZERO

var _captured := false
var _captured_any := false
var _capture_error := ""
var _visible_splats_at_capture := 0
var _center_luma_variance := 0.0
var _center_mean_color := Color(0.0, 0.0, 0.0, 1.0)
var _corner_mean_color := Color(0.0, 0.0, 0.0, 1.0)
var _max_loaded_chunks := 0
var _max_visible_chunks := 0
var _max_total_chunks := 0
var _max_streaming_visible_splats := 0
var _max_chunks_loaded_this_frame := 0
var _runtime_capacity_zero_seen := false
var _runtime_buffer_invalid_seen := false
var _upload_frame_cap_hit_seen := false
var _upload_bandwidth_cap_hit_seen := false
var _chunk_load_cap_hit_seen := false
var _queue_pressure_active_seen := false
var _frame_of_capture := -1
var _prev_streaming_settings := {}
var _prev_cached_render_reuse_enabled := true
var _had_prev_cached_render_reuse_value := false
var _capture_deadline_s := 0.0
var _capture_start_s := 0.0
var _capture_window_opened := false
var _capture_attempts := 0
var _streaming_data_source_seen := false
var _last_data_source := ""
var _global_luma_variance := 0.0
var _had_streaming_monitor_sample := false
var _had_renderer_streaming_stats := false
var _readiness_reached := false

func _ready():
	test_name = "Streaming Visual Smoke"
	test_duration = 5.0
	warmup_frames = 10
	super._ready()

	world_node = get_node_or_null("World")
	camera = get_node_or_null("Camera3D")

func _is_headless_runtime() -> bool:
	return OS.has_feature("headless") or DisplayServer.get_name() == "headless"

func _on_test_start():
	_captured = false
	_captured_any = false
	_capture_error = ""
	_visible_splats_at_capture = 0
	_center_luma_variance = 0.0
	_center_mean_color = Color(0.0, 0.0, 0.0, 1.0)
	_corner_mean_color = Color(0.0, 0.0, 0.0, 1.0)
	_max_loaded_chunks = 0
	_max_visible_chunks = 0
	_max_total_chunks = 0
	_max_streaming_visible_splats = 0
	_max_chunks_loaded_this_frame = 0
	_runtime_capacity_zero_seen = false
	_runtime_buffer_invalid_seen = false
	_upload_frame_cap_hit_seen = false
	_upload_bandwidth_cap_hit_seen = false
	_chunk_load_cap_hit_seen = false
	_queue_pressure_active_seen = false
	_frame_of_capture = -1
	_capture_attempts = 0
	_streaming_data_source_seen = false
	_last_data_source = ""
	_global_luma_variance = 0.0
	_had_streaming_monitor_sample = false
	_had_renderer_streaming_stats = false
	_readiness_reached = false
	_capture_deadline_s = 0.0
	_capture_start_s = Time.get_ticks_msec() / 1000.0
	_capture_window_opened = false

	if _is_headless_runtime():
		result_metrics["skipped"] = true
		_test_result = true
		_test_message = "[QA_SKIP] Requires non-headless viewport."
		_finish_test()
		return

	if _prev_streaming_settings.is_empty():
		_prev_streaming_settings["rendering/gaussian_splatting/streaming/enabled"] = ProjectSettings.get_setting(
			"rendering/gaussian_splatting/streaming/enabled", false
		)
		_prev_streaming_settings["rendering/gaussian_splatting/instance_pipeline/enabled"] = ProjectSettings.get_setting(
			"rendering/gaussian_splatting/instance_pipeline/enabled", false
		)
		_prev_streaming_settings["rendering/gaussian_splatting/streaming/chunk_frustum_culling_enabled"] = ProjectSettings.get_setting(
			"rendering/gaussian_splatting/streaming/chunk_frustum_culling_enabled", true
		)

	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/enabled", true)
	ProjectSettings.set_setting("rendering/gaussian_splatting/instance_pipeline/enabled", true)
	ProjectSettings.set_setting("rendering/gaussian_splatting/streaming/chunk_frustum_culling_enabled", false)

	if world_node != null:
		world_node.clear_world()
		world_node.apply_world()
		if world_node.has_method("get_renderer"):
			_renderer = world_node.get_renderer()
			if _renderer != null and _renderer.has_method("is_cached_render_reuse_enabled"):
				_prev_cached_render_reuse_enabled = bool(_renderer.is_cached_render_reuse_enabled())
				_had_prev_cached_render_reuse_value = true
			if _renderer != null and _renderer.has_method("set_cached_render_reuse_enabled"):
				_renderer.set_cached_render_reuse_enabled(false)
			if _renderer != null and _renderer.has_method("set_debug_pipeline_trace_enabled"):
				_renderer.set_debug_pipeline_trace_enabled(true)
			if _renderer != null and _renderer.has_method("set_debug_binning_counters_enabled"):
				_renderer.set_debug_binning_counters_enabled(true)

	_focus_point = _resolve_focus_point()
	_far_camera_world_position = _focus_point + far_position
	_near_camera_world_position = _focus_point + near_position
	result_metrics["focus_point"] = _focus_point

	if camera != null:
		camera.global_position = _far_camera_world_position
		camera.look_at(_focus_point, Vector3.UP)

	capture_timeout_seconds = max(capture_timeout_seconds, 0.5)
	capture_delay_seconds = max(capture_delay_seconds, 0.0)
	_capture_deadline_s = 0.0
	test_duration = max(test_duration, capture_delay_seconds + capture_timeout_seconds + 1.0)

func _on_test_frame(_delta: float):
	if _captured:
		return

	var now_s = Time.get_ticks_msec() / 1000.0

	if frame_count == move_delay_frames and camera != null:
		camera.global_position = _near_camera_world_position
		camera.look_at(_focus_point, Vector3.UP)

	var total_chunks = get_custom_monitor_value("gaussian_splatting/streaming_total_chunks")
	var loaded_chunks = get_custom_monitor_value("gaussian_splatting/streaming_loaded_chunks")
	var visible_chunks = get_custom_monitor_value("gaussian_splatting/streaming_visible_chunks")
	var streaming_visible_splats = get_custom_monitor_value("gaussian_splatting/streaming_visible_count")
	var chunks_loaded_this_frame = get_custom_monitor_value("gaussian_splatting/streaming_chunks_loaded_this_frame")
	var runtime_capacity_zero = get_custom_monitor_value("gaussian_splatting/streaming_runtime_capacity_zero")
	var runtime_buffer_invalid = get_custom_monitor_value("gaussian_splatting/streaming_runtime_buffer_invalid")
	var upload_frame_cap_hit = get_custom_monitor_value("gaussian_splatting/streaming_upload_frame_cap_hit")
	var upload_bandwidth_cap_hit = get_custom_monitor_value("gaussian_splatting/streaming_upload_bandwidth_cap_hit")
	var chunk_load_cap_hit = get_custom_monitor_value("gaussian_splatting/streaming_chunk_load_cap_hit")
	var queue_pressure_active = get_custom_monitor_value("gaussian_splatting/streaming_queue_pressure_active")
	var streaming_monitor_ready = get_custom_monitor_value("gaussian_splatting/streaming_monitor_ready")
	var sort_route_monitor = get_custom_monitor_value("gaussian_splatting/sort_route_uid")
	var monitor_ready = streaming_monitor_ready != null and int(streaming_monitor_ready) > 0
	var render_stats = _read_renderer_stats()
	var route_uid = str(render_stats.get("route_uid", ""))
	var sort_route_uid = str(render_stats.get("sort_route_uid", ""))
	var sort_route_uid_monitor = "" if sort_route_monitor == null else str(sort_route_monitor)
	if sort_route_uid.is_empty() and not sort_route_uid_monitor.is_empty():
		sort_route_uid = sort_route_uid_monitor
	var sort_route_monitor_matches_stats = sort_route_uid_monitor.is_empty() or sort_route_uid_monitor == sort_route_uid
	var stats_visible_splats = int(render_stats.get("visible_splats", -1))
	var stage_cull_status = str(render_stats.get("stage_cull_status", ""))
	var stage_sort_status = str(render_stats.get("stage_sort_status", ""))
	var stage_raster_status = str(render_stats.get("stage_raster_status", ""))
	var stage_cull_reason = str(render_stats.get("stage_cull_reason", ""))
	var stage_sort_reason = str(render_stats.get("stage_sort_reason", ""))
	var stage_raster_reason = str(render_stats.get("stage_raster_reason", ""))
	var cull_gpu_visible_count = int(render_stats.get("cull_gpu_visible_count", 0))
	var sorted_splats = int(render_stats.get("sorted_splats", 0))
	var stage_sort_input_count = int(render_stats.get("stage_sort_input_count", 0))
	var cache_reuse_enabled = bool(render_stats.get("cached_render_reuse_enabled", false))
	if monitor_ready or total_chunks != null or loaded_chunks != null or visible_chunks != null or streaming_visible_splats != null:
		_had_streaming_monitor_sample = true

	if total_chunks != null:
		_max_total_chunks = max(_max_total_chunks, int(total_chunks))
	if loaded_chunks != null:
		_max_loaded_chunks = max(_max_loaded_chunks, int(loaded_chunks))
	if visible_chunks != null:
		_max_visible_chunks = max(_max_visible_chunks, int(visible_chunks))
	if streaming_visible_splats != null:
		_max_streaming_visible_splats = max(_max_streaming_visible_splats, int(streaming_visible_splats))
	if chunks_loaded_this_frame != null:
		_max_chunks_loaded_this_frame = max(_max_chunks_loaded_this_frame, int(chunks_loaded_this_frame))
	if runtime_capacity_zero != null and int(runtime_capacity_zero) > 0:
		_runtime_capacity_zero_seen = true
	if runtime_buffer_invalid != null and int(runtime_buffer_invalid) > 0:
		_runtime_buffer_invalid_seen = true
	if upload_frame_cap_hit != null and int(upload_frame_cap_hit) > 0:
		_upload_frame_cap_hit_seen = true
	if upload_bandwidth_cap_hit != null and int(upload_bandwidth_cap_hit) > 0:
		_upload_bandwidth_cap_hit_seen = true
	if chunk_load_cap_hit != null and int(chunk_load_cap_hit) > 0:
		_chunk_load_cap_hit_seen = true
	if queue_pressure_active != null and int(queue_pressure_active) > 0:
		_queue_pressure_active_seen = true
	if render_stats.has("cull_static_chunk_total"):
		_had_renderer_streaming_stats = true
		_max_total_chunks = max(_max_total_chunks, int(render_stats.get("cull_static_chunk_total", 0)))
	if render_stats.has("cull_visible_static_chunks"):
		_had_renderer_streaming_stats = true
		var cull_visible_chunks = int(render_stats.get("cull_visible_static_chunks", 0))
		_max_visible_chunks = max(_max_visible_chunks, cull_visible_chunks)
		# Lower-bound fallback when explicit loaded-chunk monitor is unavailable.
		_max_loaded_chunks = max(_max_loaded_chunks, cull_visible_chunks)
	if render_stats.has("data_source"):
		_had_renderer_streaming_stats = true
		_last_data_source = str(render_stats.get("data_source", ""))
		if _last_data_source.findn("stream") != -1:
			_streaming_data_source_seen = true

	result_metrics["total_chunks_max"] = _max_total_chunks
	result_metrics["loaded_chunks_max"] = _max_loaded_chunks
	result_metrics["visible_chunks_max"] = _max_visible_chunks
	result_metrics["streaming_visible_splats_max"] = _max_streaming_visible_splats
	result_metrics["chunks_loaded_this_frame_max"] = _max_chunks_loaded_this_frame
	result_metrics["runtime_capacity_zero_seen"] = _runtime_capacity_zero_seen
	result_metrics["runtime_buffer_invalid_seen"] = _runtime_buffer_invalid_seen
	result_metrics["upload_frame_cap_hit_seen"] = _upload_frame_cap_hit_seen
	result_metrics["upload_bandwidth_cap_hit_seen"] = _upload_bandwidth_cap_hit_seen
	result_metrics["chunk_load_cap_hit_seen"] = _chunk_load_cap_hit_seen
	result_metrics["queue_pressure_active_seen"] = _queue_pressure_active_seen
	result_metrics["streaming_data_source_seen"] = _streaming_data_source_seen
	result_metrics["renderer_data_source"] = _last_data_source
	result_metrics["streaming_monitor_sample_seen"] = _had_streaming_monitor_sample
	result_metrics["streaming_monitor_ready"] = monitor_ready
	result_metrics["renderer_streaming_stats_seen"] = _had_renderer_streaming_stats
	result_metrics["route_uid"] = route_uid
	result_metrics["sort_route_uid"] = sort_route_uid
	result_metrics["sort_route_uid_monitor"] = sort_route_uid_monitor
	result_metrics["sort_route_uid_monitor_matches_stats"] = sort_route_monitor_matches_stats
	result_metrics["stage_cull_status"] = stage_cull_status
	result_metrics["stage_sort_status"] = stage_sort_status
	result_metrics["stage_raster_status"] = stage_raster_status
	result_metrics["stage_cull_reason"] = stage_cull_reason
	result_metrics["stage_sort_reason"] = stage_sort_reason
	result_metrics["stage_raster_reason"] = stage_raster_reason
	result_metrics["cull_gpu_visible_count"] = cull_gpu_visible_count
	result_metrics["sorted_splats"] = sorted_splats
	result_metrics["stage_sort_input_count"] = stage_sort_input_count
	result_metrics["cached_render_reuse_enabled"] = cache_reuse_enabled

	var visible_splats = _read_visible_splats()
	result_metrics["visible_splats"] = visible_splats
	if frame_count % 30 == 0:
		var renderer_id := -1
		var has_visible_method := false
		if _renderer != null:
			renderer_id = int(_renderer.get_instance_id())
			has_visible_method = _renderer.has_method("get_visible_splat_count")
		var stats_frame = int(render_stats.get("frame_count", -1))
		var stage_valid = bool(render_stats.get("stage_metrics_valid", false))
		print("[QA:Streaming Visual Smoke][diag] frame=%d stats_frame=%d stage_valid=%s renderer_id=%d has_visible_method=%s visible_method=%d visible_stats=%d route=%s sort_route=%s stage=%s/%s/%s sorted=%d sort_input=%d stream_visible=%d vis_chunks=%d loaded_chunks=%d total_chunks=%d" % [
			frame_count,
			stats_frame,
			stage_valid,
			renderer_id,
			has_visible_method,
			visible_splats,
			stats_visible_splats,
			route_uid,
			sort_route_uid,
			stage_cull_status,
			stage_sort_status,
			stage_raster_status,
			sorted_splats,
			stage_sort_input_count,
			_max_streaming_visible_splats,
			_max_visible_chunks,
			_max_loaded_chunks,
			_max_total_chunks,
		])

	# Start capture once either delay budget is satisfied to avoid low-FPS stalls.
	if frame_count < capture_delay_frames and (now_s - _capture_start_s) < capture_delay_seconds:
		return

	if not _capture_window_opened:
		_capture_window_opened = true
		_capture_deadline_s = now_s + capture_timeout_seconds

	var has_visibility_signal = visible_splats > 0 or _max_visible_chunks > 0 or _max_streaming_visible_splats > 0
	if not has_visibility_signal:
		if now_s < _capture_deadline_s:
			return
		_capture_error = "No visible splats before capture timeout (route=%s sort_route=%s stage=%s/%s/%s reasons=%s|%s|%s cull_gpu=%d sorted=%d sort_input=%d chunks=%d/%d vis_chunks=%d stream_visible=%d loaded_this_frame_max=%d source=%s cache_reuse=%s cap0=%s buf_invalid=%s load_cap=%s upload_cap=%s bw_cap=%s q_pressure=%s)" % [
			route_uid,
			sort_route_uid,
			stage_cull_status,
			stage_sort_status,
			stage_raster_status,
			stage_cull_reason,
			stage_sort_reason,
			stage_raster_reason,
			cull_gpu_visible_count,
			sorted_splats,
			stage_sort_input_count,
			_max_loaded_chunks,
			_max_total_chunks,
			_max_visible_chunks,
			_max_streaming_visible_splats,
			_max_chunks_loaded_this_frame,
			_last_data_source,
			cache_reuse_enabled,
			_runtime_capacity_zero_seen,
			_runtime_buffer_invalid_seen,
			_chunk_load_cap_hit_seen,
			_upload_frame_cap_hit_seen,
			_upload_bandwidth_cap_hit_seen,
			_queue_pressure_active_seen,
		]
		_captured = true
		_finish_test()
		return

	var image = capture_viewport()
	if image == null:
		if now_s < _capture_deadline_s:
			return
		_capture_error = "Viewport capture unavailable before timeout"
		_captured = true
		_finish_test()
		return

	var center = Vector2i(image.get_width() / 2, image.get_height() / 2)
	var center_stats = _compute_patch_stats(image, center, patch_half_extent)
	_center_mean_color = center_stats["mean_color"]
	_center_luma_variance = center_stats["luma_variance"]
	_global_luma_variance = _compute_global_luma_variance(image, 8)
	_corner_mean_color = _compute_corner_mean_color(image, patch_half_extent)
	_visible_splats_at_capture = max(visible_splats, _max_streaming_visible_splats)
	_frame_of_capture = frame_count
	_capture_attempts += 1
	_captured_any = true

	result_metrics["capture_frame"] = _frame_of_capture
	result_metrics["center_luma_variance"] = _center_luma_variance
	result_metrics["global_luma_variance"] = _global_luma_variance
	result_metrics["center_color"] = _center_mean_color
	result_metrics["corner_color"] = _corner_mean_color
	result_metrics["capture_attempts"] = _capture_attempts

	var center_to_corner_delta = _color_distance(_center_mean_color, _corner_mean_color)
	result_metrics["center_to_corner_delta"] = center_to_corner_delta
	var readiness = _evaluate_readiness(_visible_splats_at_capture, center_to_corner_delta)
	_readiness_reached = bool(readiness.get("ready", false))
	result_metrics["streaming_signal_ok"] = bool(readiness.get("streaming_signal_ok", false))
	result_metrics["chunk_signal_ok"] = bool(readiness.get("chunk_signal_ok", false))
	result_metrics["monitor_stats_available"] = bool(readiness.get("monitor_stats_available", false))
	result_metrics["monitor_signal_ok"] = bool(readiness.get("monitor_signal_ok", false))
	result_metrics["visibility_ok"] = bool(readiness.get("visibility_ok", false))
	result_metrics["visual_ok"] = bool(readiness.get("visual_ok", false))
	if not _readiness_reached:
		if now_s < _capture_deadline_s:
			return
		_capture_error = "Streaming/visual readiness not reached (variance=%.5f global=%.5f delta=%.3f chunks=%d/%d source=%s stream_visible=%d monitors=%s stats=%s)" % [
			_center_luma_variance,
			_global_luma_variance,
			center_to_corner_delta,
			_max_loaded_chunks,
			_max_total_chunks,
			_last_data_source,
			_max_streaming_visible_splats,
			_had_streaming_monitor_sample,
			_had_renderer_streaming_stats,
		]
		_captured = true
		_finish_test()
		return

	_captured = true
	_finish_test()

func _on_test_complete():
	for key in _prev_streaming_settings.keys():
		ProjectSettings.set_setting(key, _prev_streaming_settings[key])
	_prev_streaming_settings.clear()
	if _renderer != null and _had_prev_cached_render_reuse_value and _renderer.has_method("set_cached_render_reuse_enabled"):
		_renderer.set_cached_render_reuse_enabled(_prev_cached_render_reuse_enabled)
	_had_prev_cached_render_reuse_value = false

	if bool(result_metrics.get("skipped", false)):
		_test_result = true
		if _test_message.is_empty():
			_test_message = "[QA_SKIP] Requires non-headless viewport."
		return

	if not _capture_error.is_empty():
		_test_result = false
		_test_message = _capture_error
		return

	if not _captured_any:
		_test_result = false
		_test_message = "No capture"
		return

	var center_to_corner_delta = _color_distance(_center_mean_color, _corner_mean_color)
	result_metrics["center_to_corner_delta"] = center_to_corner_delta
	result_metrics["visible_splats_at_capture"] = _visible_splats_at_capture

	var readiness = _evaluate_readiness(_visible_splats_at_capture, center_to_corner_delta)
	_readiness_reached = bool(readiness.get("ready", false))
	result_metrics["streaming_signal_ok"] = bool(readiness.get("streaming_signal_ok", false))
	result_metrics["chunk_signal_ok"] = bool(readiness.get("chunk_signal_ok", false))
	result_metrics["monitor_stats_available"] = bool(readiness.get("monitor_stats_available", false))
	result_metrics["monitor_signal_ok"] = bool(readiness.get("monitor_signal_ok", false))
	result_metrics["visibility_ok"] = bool(readiness.get("visibility_ok", false))
	result_metrics["visual_ok"] = bool(readiness.get("visual_ok", false))
	_test_result = _readiness_reached
	_test_message = "visible=%d chunks(loaded/total)=%d/%d variance=%.5f global=%.5f delta=%.3f source=%s stream_visible=%d attempts=%d monitor_signal=%s" % [
		_visible_splats_at_capture,
		_max_loaded_chunks,
		_max_total_chunks,
		_center_luma_variance,
		_global_luma_variance,
		center_to_corner_delta,
		_last_data_source,
		_max_streaming_visible_splats,
		_capture_attempts,
		bool(readiness.get("monitor_signal_ok", false)),
	]

func _read_visible_splats() -> int:
	_refresh_renderer_handle()
	if _renderer == null:
		return 0

	if _renderer.has_method("get_visible_splat_count"):
		return int(_renderer.get_visible_splat_count())

	if _renderer.has_method("get_render_stats"):
		var stats = _renderer.get_render_stats()
		if stats is Dictionary:
			if stats.has("visible_splats"):
				return int(stats.get("visible_splats", 0))
			if stats.has("visible_after_culling"):
				return int(stats.get("visible_after_culling", 0))
	return 0

func _read_renderer_stats() -> Dictionary:
	_refresh_renderer_handle()
	if _renderer == null:
		return {}
	if not _renderer.has_method("get_render_stats"):
		return {}
	var stats = _renderer.get_render_stats()
	if stats is Dictionary:
		return stats
	return {}

func _refresh_renderer_handle() -> void:
	if world_node == null or not world_node.has_method("get_renderer"):
		return
	var latest_renderer = world_node.get_renderer()
	if latest_renderer != null:
		_renderer = latest_renderer

func _resolve_focus_point() -> Vector3:
	if world_node == null or not world_node.has_method("get_world"):
		return Vector3.ZERO
	var world_res = world_node.get_world()
	if world_res == null or not world_res.has_method("get_bounds"):
		return Vector3.ZERO
	var bounds = world_res.get_bounds()
	if bounds is AABB and bounds.has_volume():
		return bounds.get_center()
	return Vector3.ZERO

func _compute_patch_stats(image: Image, center: Vector2i, half_extent: int) -> Dictionary:
	var min_x = max(0, center.x - half_extent)
	var max_x = min(image.get_width() - 1, center.x + half_extent)
	var min_y = max(0, center.y - half_extent)
	var max_y = min(image.get_height() - 1, center.y + half_extent)

	var accum = Color(0.0, 0.0, 0.0, 1.0)
	var luma_sum = 0.0
	var luma_sq_sum = 0.0
	var count = 0

	for y in range(min_y, max_y + 1):
		for x in range(min_x, max_x + 1):
			var c = image.get_pixel(x, y)
			accum += c
			var luma = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
			luma_sum += luma
			luma_sq_sum += luma * luma
			count += 1

	if count <= 0:
		return {
			"mean_color": Color(0.0, 0.0, 0.0, 1.0),
			"luma_variance": 0.0,
		}

	var mean_color = accum * (1.0 / float(count))
	var mean_luma = luma_sum / float(count)
	var variance = max(0.0, (luma_sq_sum / float(count)) - (mean_luma * mean_luma))

	return {
		"mean_color": mean_color,
		"luma_variance": variance,
	}

func _compute_corner_mean_color(image: Image, half_extent: int) -> Color:
	var width = image.get_width()
	var height = image.get_height()
	var corners = [
		Vector2i(half_extent, half_extent),
		Vector2i(width - 1 - half_extent, half_extent),
		Vector2i(half_extent, height - 1 - half_extent),
		Vector2i(width - 1 - half_extent, height - 1 - half_extent),
	]

	var accum = Color(0.0, 0.0, 0.0, 1.0)
	for point in corners:
		var stats = _compute_patch_stats(image, point, half_extent)
		accum += stats["mean_color"]

	return accum * (1.0 / float(corners.size()))

func _color_distance(a: Color, b: Color) -> float:
	var dr = a.r - b.r
	var dg = a.g - b.g
	var db = a.b - b.b
	return sqrt(dr * dr + dg * dg + db * db)

func _compute_global_luma_variance(image: Image, sample_stride: int = 8) -> float:
	var width = image.get_width()
	var height = image.get_height()
	if width <= 0 or height <= 0:
		return 0.0
	var stride = max(1, sample_stride)
	var luma_sum = 0.0
	var luma_sq_sum = 0.0
	var count = 0
	for y in range(0, height, stride):
		for x in range(0, width, stride):
			var c = image.get_pixel(x, y)
			var luma = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
			luma_sum += luma
			luma_sq_sum += luma * luma
			count += 1
	if count <= 0:
		return 0.0
	var mean_luma = luma_sum / float(count)
	return max(0.0, (luma_sq_sum / float(count)) - (mean_luma * mean_luma))

func _evaluate_readiness(visible_splats: int, center_to_corner_delta: float) -> Dictionary:
	var streaming_signal_ok = _streaming_data_source_seen or _max_streaming_visible_splats > 0
	var chunk_signal_ok = _max_total_chunks > 0 and (_max_loaded_chunks > 0 or _max_visible_chunks > 0)
	var monitor_stats_available = _had_streaming_monitor_sample or _had_renderer_streaming_stats
	var monitor_signal_ok = streaming_signal_ok or chunk_signal_ok
	var visibility_ok = visible_splats > 0 or _max_visible_chunks > 0 or _max_streaming_visible_splats > 0
	var visual_ok = _capture_attempts >= min_capture_attempts and (
		_center_luma_variance >= min_center_luma_variance or
		center_to_corner_delta >= min_center_to_corner_delta or
		_global_luma_variance >= min_global_luma_variance
	)
	var ready = visibility_ok and visual_ok and ((not monitor_stats_available) or monitor_signal_ok)
	return {
		"ready": ready,
		"streaming_signal_ok": streaming_signal_ok,
		"chunk_signal_ok": chunk_signal_ok,
		"monitor_stats_available": monitor_stats_available,
		"monitor_signal_ok": monitor_signal_ok,
		"visibility_ok": visibility_ok,
		"visual_ok": visual_ok,
	}
