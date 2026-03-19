extends Node3D

# Diagnostic test for #797: Compute rasterizer tile artifacts at distance
# This scene moves the camera through multiple distances and logs overflow statistics
# to identify if per-tile overflow correlates with visual artifacts.

const TEST_DISTANCES := [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
const SAMPLES_PER_DISTANCE := 60  # ~1 second at 60fps
const LOG_INTERVAL := 10  # Log every N samples
const WARMUP_FRAMES := 60  # Wait for streaming to initialize

var _splat_node: GaussianSplatNode3D
var _camera: Camera3D
var _renderer: GaussianSplatRenderer

var _current_distance_idx := 0
var _samples_at_distance := 0
var _frame_count := 0
var _warmup_count := 0
var _warming_up := true
var _test_started := false

# Accumulated stats per distance (use index as key to avoid float issues)
var _distance_stats := []

func _ready() -> void:
	OS.set_low_processor_usage_mode(false)
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

	# Find or create splat node
	_splat_node = get_node_or_null("GaussianSplatNode3D")
	if not _splat_node:
		_splat_node = GaussianSplatNode3D.new()
		_splat_node.name = "GaussianSplatNode3D"
		add_child(_splat_node)

	# Always try to load the asset
	var asset = load("res://tests/fixtures/test_splats.ply")
	if asset:
		print("[DIAG-797] Loaded synthetic fixture asset")
		_splat_node.splat_asset = asset
	else:
		push_error("[DIAG-797] Missing synthetic fixture asset: res://tests/fixtures/test_splats.ply")
		get_tree().quit()
		return

	_camera = get_node_or_null("Camera3D")
	if not _camera:
		_camera = Camera3D.new()
		_camera.name = "Camera3D"
		_camera.current = true
		add_child(_camera)

	# Initialize stats tracking as array
	for i in TEST_DISTANCES.size():
		_distance_stats.append({
			"distance": TEST_DISTANCES[i],
			"samples": 0,
			"overflow_splats_clamped_total": 0,
			"overflow_splats_clamped_max": 0,
			"distance_cull_reject_total": 0,
			"distance_cull_reject_max": 0,
			"radius_reject_total": 0,
			"radius_reject_max": 0,
			"visible_splats_total": 0,
			"fps_total": 0.0,
		})

	print("[DIAG-797] ========================================")
	print("[DIAG-797] Distance Artifact Diagnostic Test")
	print("[DIAG-797] Testing %d distances: %s" % [TEST_DISTANCES.size(), TEST_DISTANCES])
	print("[DIAG-797] %d samples per distance, %d warmup frames" % [SAMPLES_PER_DISTANCE, WARMUP_FRAMES])
	print("[DIAG-797] ========================================")
	print("[DIAG-797] Warming up (waiting for streaming)...")

	_set_camera_distance(TEST_DISTANCES[0])


func _set_camera_distance(distance: float) -> void:
	# Position camera at distance, looking at origin
	_camera.global_position = Vector3(0, distance * 0.3, distance)
	_camera.look_at(Vector3.ZERO, Vector3.UP)
	print("[DIAG-797] Moving camera to distance: %.1f" % distance)


func _process(delta: float) -> void:
	_frame_count += 1

	# Get renderer lazily (may take a few frames to be available)
	if not _renderer:
		_renderer = _splat_node.get_renderer()
		if _renderer:
			print("[DIAG-797] Renderer acquired")
			_renderer.set_debug_pipeline_trace_enabled(true)
			_renderer.set_debug_binning_counters_enabled(true)

			# PROPER FIX: Enable subpixel splat culling
			# This should filter out splats that project to < 0.5 pixels at distance
			_renderer.set_tiny_splat_screen_radius(0.5)  # Cull splats < 0.5 pixels

			# Disable distance culling to test subpixel culling alone
			_renderer.set_distance_cull_enabled(false)

			print("[DIAG-797] Settings:")
			print("[DIAG-797]   tiny_splat_screen_radius: %.2f pixels" % _renderer.get_tiny_splat_screen_radius())
			print("[DIAG-797]   distance_cull_enabled: %s" % _renderer.is_distance_cull_enabled())
		return

	# Warmup phase - wait for streaming to initialize
	if _warming_up:
		_warmup_count += 1
		var visible = _renderer.get_visible_splat_count()
		if _warmup_count % 10 == 0:
			var trace = _renderer.get_pipeline_trace_snapshot()
			var route = trace.get("route_uid", "N/A") if trace else "N/A"
			print("[DIAG-797] Warmup %d/%d visible=%d route=%s" % [_warmup_count, WARMUP_FRAMES, visible, route])

		# End warmup when we have data or timeout
		if _warmup_count >= WARMUP_FRAMES or visible > 0:
			_warming_up = false
			_test_started = true
			print("[DIAG-797] Warmup complete, starting test (visible=%d)" % visible)
		return

	if _current_distance_idx >= TEST_DISTANCES.size():
		return

	_samples_at_distance += 1

	var stats = _distance_stats[_current_distance_idx]

	# Collect diagnostics
	var binning_counters = _renderer.get_binning_debug_counters()
	var visible_count = _renderer.get_visible_splat_count()
	var fps = 1.0 / delta if delta > 0 else 0.0

	# Extract counters
	var overflow_clamped = 0
	var distance_cull_reject = 0
	var radius_reject = 0
	if binning_counters:
		if binning_counters.has("overflow_splats_clamped"):
			overflow_clamped = binning_counters["overflow_splats_clamped"]
		if binning_counters.has("distance_cull_reject"):
			distance_cull_reject = binning_counters["distance_cull_reject"]
		if binning_counters.has("radius_reject"):
			radius_reject = binning_counters["radius_reject"]

	# Accumulate stats
	stats["samples"] += 1
	stats["overflow_splats_clamped_total"] += overflow_clamped
	stats["overflow_splats_clamped_max"] = max(stats["overflow_splats_clamped_max"], overflow_clamped)
	stats["distance_cull_reject_total"] += distance_cull_reject
	stats["distance_cull_reject_max"] = max(stats["distance_cull_reject_max"], distance_cull_reject)
	stats["radius_reject_total"] += radius_reject
	stats["radius_reject_max"] = max(stats["radius_reject_max"], radius_reject)
	stats["visible_splats_total"] += visible_count
	stats["fps_total"] += fps

	# Periodic logging
	if _samples_at_distance % LOG_INTERVAL == 0:
		var trace = _renderer.get_pipeline_trace_snapshot()
		var route_uid = trace.get("route_uid", "N/A") if trace else "N/A"
		var data_flow: Dictionary = trace.get("data_flow", {}) if trace else {}
		var recent_window: Dictionary = data_flow.get("recent_window", {})
		var recent_frames := int(recent_window.get("frames_recorded", 0))
		var recent_pack_sh := int(recent_window.get("pack_sh_samples", 0))
		var recent_pack_range := int(recent_window.get("pack_range_calls", 0))

		# NEW: Extract GPU-side debug counters for tiny_splat investigation
		var tiny_param_px = binning_counters.get("tiny_splat_param_px", -1.0) if binning_counters else -1.0
		var min_allowed_px = binning_counters.get("min_allowed_radius_px", -1.0) if binning_counters else -1.0
		var min_radius_min_px = binning_counters.get("min_radius_min_px", -1.0) if binning_counters else -1.0

		print("[DIAG-797] dist=%.0f sample=%d overflow=%d radius_cull=%d visible=%d fps=%.0f" % [
			stats["distance"], _samples_at_distance, overflow_clamped, radius_reject, visible_count, fps
		])
		print("[DIAG-797]   GPU: tiny_param=%.3f min_allowed=%.3f min_radius_min=%.3f" % [
			tiny_param_px, min_allowed_px, min_radius_min_px
		])
		print("[DIAG-797]   TRACE: route=%s recent_frames=%d recent_pack_sh=%d recent_pack_range=%d" % [
			route_uid, recent_frames, recent_pack_sh, recent_pack_range
		])

	# Check if done with this distance
	if _samples_at_distance >= SAMPLES_PER_DISTANCE:
		_report_distance_summary(stats)

		_current_distance_idx += 1
		_samples_at_distance = 0

		if _current_distance_idx < TEST_DISTANCES.size():
			_set_camera_distance(TEST_DISTANCES[_current_distance_idx])
		else:
			_report_final_summary()

			# Dump final pipeline trace
			var trace_path = "user://diag_797_trace.json"
			_renderer.dump_pipeline_trace_to_file(trace_path)
			print("[DIAG-797] Pipeline trace saved to: %s" % trace_path)

			get_tree().quit()


func _report_distance_summary(stats: Dictionary) -> void:
	var samples = stats["samples"]
	if samples == 0:
		return

	var distance = stats["distance"]
	var avg_overflow = stats["overflow_splats_clamped_total"] / float(samples)
	var max_overflow = stats["overflow_splats_clamped_max"]
	var avg_radius_reject = stats["radius_reject_total"] / float(samples)
	var max_radius_reject = stats["radius_reject_max"]
	var avg_visible = stats["visible_splats_total"] / float(samples)
	var avg_fps = stats["fps_total"] / float(samples)

	print("[DIAG-797] ----------------------------------------")
	print("[DIAG-797] Distance %.0f Summary:" % distance)
	print("[DIAG-797]   Samples: %d" % samples)
	print("[DIAG-797]   Overflow clamped: avg=%.1f max=%d" % [avg_overflow, max_overflow])
	print("[DIAG-797]   Radius reject (subpixel cull): avg=%.0f max=%d" % [avg_radius_reject, max_radius_reject])
	print("[DIAG-797]   Visible splats: avg=%.0f" % avg_visible)
	print("[DIAG-797]   FPS: avg=%.1f" % avg_fps)

	if max_overflow > 0:
		print("[DIAG-797]   WARNING: OVERFLOW DETECTED - May cause artifacts!")
	if max_radius_reject > 0:
		print("[DIAG-797]   Subpixel culling active: %d splats culled" % max_radius_reject)


func _report_final_summary() -> void:
	print("[DIAG-797] ========================================")
	print("[DIAG-797] FINAL SUMMARY")
	print("[DIAG-797] ========================================")

	var has_any_overflow = false
	var has_any_radius_cull = false
	var total_radius_culled = 0

	for stats in _distance_stats:
		var samples = stats["samples"]
		if samples == 0:
			continue

		var distance = stats["distance"]
		var avg_overflow = stats["overflow_splats_clamped_total"] / float(samples)
		var max_overflow = stats["overflow_splats_clamped_max"]
		var avg_radius_reject = stats["radius_reject_total"] / float(samples)
		var max_radius_reject = stats["radius_reject_max"]
		var overflow_marker = " *OVERFLOW*" if max_overflow > 0 else ""
		var cull_marker = " (subpixel:%d)" % max_radius_reject if max_radius_reject > 0 else ""

		if max_overflow > 0:
			has_any_overflow = true
		if max_radius_reject > 0:
			has_any_radius_cull = true
			total_radius_culled += stats["radius_reject_total"]

		print("[DIAG-797] dist=%6.0f | overflow: avg=%5.1f max=%5d | visible: %7.0f | radius_cull: %6.0f%s%s" % [
			distance,
			avg_overflow,
			max_overflow,
			stats["visible_splats_total"] / float(samples),
			avg_radius_reject,
			overflow_marker,
			cull_marker
		])

	print("[DIAG-797] ========================================")

	if has_any_radius_cull:
		print("[DIAG-797] Subpixel culling: ACTIVE (total culled: %d)" % total_radius_culled)
	else:
		print("[DIAG-797] Subpixel culling: NOT WORKING (zero splats culled!)")

	if has_any_overflow:
		print("[DIAG-797] DIAGNOSIS: Per-tile overflow detected!")
		if has_any_radius_cull:
			print("[DIAG-797] Subpixel culling is active but insufficient.")
			print("[DIAG-797] Consider increasing tiny_splat_screen_radius.")
		else:
			print("[DIAG-797] Subpixel culling is NOT working - check tiny_splat_screen_radius.")
		print("[DIAG-797] ")
		print("[DIAG-797] Potential fixes:")
		print("[DIAG-797]   1. Increase tiny_splat_screen_radius (try 1.0 or higher)")
		print("[DIAG-797]   2. Enable LOD system to reduce splat count at distance")
		print("[DIAG-797]   3. Increase SPLATS_PER_TILE if GPU memory allows")
	else:
		print("[DIAG-797] SUCCESS: No overflow detected!")
		print("[DIAG-797] Subpixel culling is effectively preventing tile overflow.")


func _input(event: InputEvent) -> void:
	if not _renderer:
		return
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_ESCAPE:
				print("[DIAG-797] Test aborted by user")
				get_tree().quit()
			KEY_H:
				# Toggle density heatmap
				var enabled = _renderer.is_debug_show_density_heatmap()
				_renderer.set_debug_show_density_heatmap(not enabled)
				print("[DIAG-797] Density heatmap: %s" % (not enabled))
			KEY_O:
				# Toggle overflow tile visualization
				var enabled = _renderer.get_debug_show_overflow_tiles()
				_renderer.set_debug_show_overflow_tiles(not enabled)
				print("[DIAG-797] Overflow tiles: %s" % (not enabled))
			KEY_G:
				# Toggle tile grid
				var enabled = _renderer.is_debug_show_tile_grid()
				_renderer.set_debug_show_tile_grid(not enabled)
				print("[DIAG-797] Tile grid: %s" % (not enabled))
