class_name GSQATest
extends Node3D
## Base class for Gaussian Splatting QA tests.
## Provides common infrastructure for warmup, measurement, and reporting.

signal test_completed(passed: bool, message: String)
signal test_progress(percent: float, status: String)

@export var test_name: String = "Unnamed Test"
@export var test_duration: float = 10.0
@export var warmup_frames: int = 30
@export var auto_start: bool = true
@export var quit_on_complete: bool = true

enum TestPhase { IDLE, WARMUP, RUNNING, COMPLETE }

var phase: TestPhase = TestPhase.IDLE
var frame_count: int = 0
var start_time: float = 0.0
var _test_result: bool = false
var _test_message: String = ""
var result_metrics: Dictionary = {}

const SSIM_WINDOW_SIZE := 11
const SSIM_SIGMA := 1.5
const SSIM_K1 := 0.01
const SSIM_K2 := 0.03
const SSIM_MAX_DIM := 256
var _ssim_kernel: PackedFloat32Array

func _ready():
	set_process(false)
	if auto_start:
		# Delay to let scene fully initialize
		await get_tree().create_timer(1.0).timeout
		if not auto_start:
			return
		if phase != TestPhase.IDLE:
			return
		start_test()

func start_test():
	if phase == TestPhase.WARMUP or phase == TestPhase.RUNNING:
		return
	phase = TestPhase.WARMUP
	frame_count = 0
	_test_result = false
	_test_message = ""
	result_metrics.clear()
	print("[QA:%s] Starting warmup (%d frames)..." % [test_name, warmup_frames])
	set_process(true)

func _process(delta: float):
	frame_count += 1

	match phase:
		TestPhase.WARMUP:
			if frame_count >= warmup_frames:
				phase = TestPhase.RUNNING
				frame_count = 0
				start_time = Time.get_ticks_msec() / 1000.0
				print("[QA:%s] Warmup complete, running test..." % test_name)
				_on_test_start()

		TestPhase.RUNNING:
			var elapsed = Time.get_ticks_msec() / 1000.0 - start_time
			var progress = elapsed / test_duration
			emit_signal("test_progress", progress, "Testing...")

			_on_test_frame(delta)

			if elapsed >= test_duration:
				_finish_test()

		TestPhase.COMPLETE:
			set_process(false)

func _finish_test():
	phase = TestPhase.COMPLETE
	_on_test_complete()
	print("[QA:%s] %s: %s" % [test_name, "PASS" if _test_result else "FAIL", _test_message])
	emit_signal("test_completed", _test_result, _test_message)

	if quit_on_complete:
		await get_tree().create_timer(2.0).timeout
		get_tree().quit(0 if _test_result else 1)

## Override in subclass - called when test phase starts
func _on_test_start():
	pass

## Override in subclass - called each frame during test
func _on_test_frame(_delta: float):
	pass

## Override in subclass - called when test duration ends
## Set _test_result and _test_message before returning
func _on_test_complete():
	_test_result = true
	_test_message = "Base test completed (override _on_test_complete)"

func get_result_metrics() -> Dictionary:
	return result_metrics

## Helper: Append a stable subset of renderer diagnostics into result metrics.
func append_renderer_diagnostics(metric_prefix: String, renderer: Object) -> void:
	if renderer == null or not renderer.has_method("get_render_stats"):
		return
	var stats = renderer.get_render_stats()
	if not (stats is Dictionary):
		return

	var prefix = metric_prefix
	if not prefix.is_empty() and not prefix.ends_with("_"):
		prefix += "_"

	var keys := [
		"route_uid",
		"sort_route_uid",
		"data_source",
		"raster_path",
		"gpu_sorter_algorithm",
		"gpu_sorter_last_sort_ms",
		"stage_metrics_valid",
		"stage_cull_status",
		"stage_sort_status",
		"stage_raster_status",
		"stage_composite_status",
		"stage_cull_reason",
		"stage_sort_reason",
		"stage_raster_reason",
		"stage_composite_reason",
		"stage_raster_cached",
		"stage_sort_sorted_count",
		"stage_sort_input_count",
		"cull_gpu_visible_count",
		"cull_visible_static_chunks",
		"cull_static_chunk_total",
		"visible_splats",
		"sorted_splats",
		"sort_cache_hits",
		"sort_cache_misses",
		"sorted_indices_preview",
		"overflow_tile_count",
		"raster_reject_index_mismatch",
		"instance_pipeline_content_generation",
		"cached_render_reuse_enabled",
	]
	for key in keys:
		if stats.has(key):
			result_metrics[prefix + key] = stats[key]

## Helper: Get renderer from a GaussianSplatNode3D child
func get_gs_renderer(node_path: NodePath) -> Object:
	var node = get_node_or_null(node_path)
	if node == null:
		return null
	if node.has_method("get_renderer"):
		return node.get_renderer()
	return null

## Helper: Capture current viewport as Image
func capture_viewport() -> Image:
	var vp = get_viewport()
	if vp == null:
		return null
	return vp.get_texture().get_image()

## Helper: Read a custom Performance monitor value (returns null if unavailable)
func get_custom_monitor_value(monitor_id: String):
	if not Performance.has_custom_monitor(monitor_id):
		return null
	return Performance.get_custom_monitor(monitor_id)

## Helper: Calculate SSIM between two images (real SSIM with gaussian window)
func calculate_ssim(img_a: Image, img_b: Image) -> float:
	if img_a == null or img_b == null:
		return 0.0
	var a = _prepare_ssim_image(img_a)
	var b = _prepare_ssim_image(img_b)
	if a == null or b == null:
		return 0.0
	if a.get_size() != b.get_size():
		return 0.0

	var width = a.get_width()
	var height = a.get_height()
	if width < SSIM_WINDOW_SIZE or height < SSIM_WINDOW_SIZE:
		return 0.0

	var luma_a = _compute_luma_buffer(a)
	var luma_b = _compute_luma_buffer(b)
	var kernel = _get_ssim_kernel()
	var radius = int(SSIM_WINDOW_SIZE / 2)

	var c1 = pow(SSIM_K1, 2)
	var c2 = pow(SSIM_K2, 2)

	var ssim_sum: float = 0.0
	var count: int = 0

	for y in range(radius, height - radius):
		for x in range(radius, width - radius):
			var mean_a = 0.0
			var mean_b = 0.0
			var k_idx = 0
			for wy in range(-radius, radius + 1):
				var row = (y + wy) * width
				for wx in range(-radius, radius + 1):
					var w = kernel[k_idx]
					k_idx += 1
					var idx = row + x + wx
					mean_a += w * luma_a[idx]
					mean_b += w * luma_b[idx]

			var var_a = 0.0
			var var_b = 0.0
			var cov = 0.0
			k_idx = 0
			for wy in range(-radius, radius + 1):
				var row = (y + wy) * width
				for wx in range(-radius, radius + 1):
					var w = kernel[k_idx]
					k_idx += 1
					var idx = row + x + wx
					var da = luma_a[idx] - mean_a
					var db = luma_b[idx] - mean_b
					var_a += w * da * da
					var_b += w * db * db
					cov += w * da * db

			var numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2)
			var denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
			if denominator > 0.0:
				ssim_sum += numerator / denominator
				count += 1

	if count == 0:
		return 0.0
	return ssim_sum / float(count)

func _prepare_ssim_image(img: Image) -> Image:
	var prepared = img.duplicate()
	if prepared == null:
		return null
	var width = prepared.get_width()
	var height = prepared.get_height()
	var max_dim = max(width, height)
	if max_dim > SSIM_MAX_DIM:
		var scale = float(SSIM_MAX_DIM) / float(max_dim)
		var new_w = max(1, int(round(width * scale)))
		var new_h = max(1, int(round(height * scale)))
		prepared.resize(new_w, new_h, Image.INTERPOLATE_BILINEAR)
	prepared.convert(Image.FORMAT_RGB8)
	return prepared

func _compute_luma_buffer(img: Image) -> PackedFloat32Array:
	var width = img.get_width()
	var height = img.get_height()
	var out = PackedFloat32Array()
	out.resize(width * height)
	var idx = 0
	for y in range(height):
		for x in range(width):
			var c = img.get_pixel(x, y)
			out[idx] = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
			idx += 1
	return out

func _get_ssim_kernel() -> PackedFloat32Array:
	if _ssim_kernel.size() == SSIM_WINDOW_SIZE * SSIM_WINDOW_SIZE:
		return _ssim_kernel
	_ssim_kernel = PackedFloat32Array()
	_ssim_kernel.resize(SSIM_WINDOW_SIZE * SSIM_WINDOW_SIZE)
	var radius = int(SSIM_WINDOW_SIZE / 2)
	var sum = 0.0
	var idx = 0
	for y in range(-radius, radius + 1):
		for x in range(-radius, radius + 1):
			var w = exp(-(float(x * x + y * y)) / (2.0 * SSIM_SIGMA * SSIM_SIGMA))
			_ssim_kernel[idx] = w
			sum += w
			idx += 1
	if sum <= 0.0:
		return _ssim_kernel
	for i in range(_ssim_kernel.size()):
		_ssim_kernel[i] /= sum
	return _ssim_kernel

## Helper: Check if FPS is within expected range
func check_fps_in_range(samples: Array[float], min_fps: float, max_fps: float) -> bool:
	if samples.is_empty():
		return false
	var avg = 0.0
	for s in samples:
		avg += s
	avg /= samples.size()
	return avg >= min_fps and avg <= max_fps

## Helper: Calculate percentile from sample array
func percentile(samples: Array[float], p: float) -> float:
	if samples.is_empty():
		return 0.0
	var sorted = samples.duplicate()
	sorted.sort()
	var idx = int((p / 100.0) * (sorted.size() - 1))
	return sorted[idx]
