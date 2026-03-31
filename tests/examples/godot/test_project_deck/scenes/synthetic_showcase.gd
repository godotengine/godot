extends Node3D

## Renders a single synthetic splat scene with configurable camera path.
## Camera modes: "orbit" (default), "flythrough" (tunnel), "sweep" (flat grid).

const BenchmarkVisualMetrics = preload("res://scripts/benchmark_visual_metrics.gd")
const BenchmarkSceneContract = preload("res://scripts/benchmark_scene_contract.gd")

@export var ply_path: String = ""
@export var duration: float = 15.0
@export var camera_mode: String = "orbit"  # "orbit", "flythrough", "sweep"

# Orbit params.
@export var orbit_radius: float = 12.0
@export var orbit_height: float = 5.0
@export var orbit_speed: float = 0.3
@export var look_at_offset: Vector3 = Vector3.ZERO

# Flythrough params (tunnel).
@export var fly_start: Vector3 = Vector3(0, 0, -10)
@export var fly_end: Vector3 = Vector3(0, 0, 10)
@export var fly_look_ahead: float = 5.0
@export var fly_speed_scale: float = 1.0

# Route policy override: -1 = use project default, 0 = resident, 1 = streaming.
@export var route_policy_override: int = -1

# Wind (applied to the splat node).
@export var enable_wind: bool = false
@export var wind_strength: float = 0.5
@export var wind_direction: Vector3 = Vector3(1, 0, 0.3)

# Sweep params (overhead grid).
@export var sweep_height: float = 8.0
@export var sweep_extent: float = 6.0

signal benchmark_scene_finished(result: Dictionary)

var _elapsed := 0.0
var _frame_count := 0
var _fps_samples: Array[float] = []
var _output_path := ""
var _capture_dir := ""
var _reference_dir := ""
var _scene_name := ""
var _lane_id := ""
var _captures: Array[Dictionary] = []
var _capture_fractions := [0.25, 0.5, 0.75]
var _captured_at: Dictionary = {}
var _pending_contract: Dictionary = {}
var _orchestrated := false

func apply_benchmark_contract(contract: Dictionary) -> void:
	_pending_contract = contract.duplicate(true)

func _ready() -> void:
	_apply_contract()
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

	# Apply route policy override before the splat node is created, so the
	# renderer sees the correct policy when it first initialises.
	if route_policy_override >= 0:
		ProjectSettings.set_setting(
			"rendering/gaussian_splatting/streaming/route_policy",
			route_policy_override
		)
		print("[SYNTHETIC] Route policy overridden to %d" % route_policy_override)
	_scene_name = BenchmarkSceneContract.scene_id_from_path(get_tree().current_scene.scene_file_path)
	if _lane_id.is_empty():
		_lane_id = _scene_name
	if _output_path.is_empty():
		_output_path = "user://%s_results.json" % _scene_name

	if ply_path.is_empty():
		push_error("[SYNTHETIC] No ply_path set")
		_finish_with_error("No ply_path set")
		return

	var splat_node = GaussianSplatNode3D.new()
	splat_node.name = "SplatNode"
	splat_node.set_ply_file_path(ply_path)
	splat_node.set_auto_load(true)
	add_child(splat_node)

	if enable_wind:
		splat_node.set("rendering/wind_override_enabled", true)
		splat_node.set("rendering/wind_enabled", true)
		splat_node.set("rendering/wind_strength", wind_strength)
		splat_node.set("rendering/wind_direction", wind_direction)

	var cam := Camera3D.new()
	cam.name = "Camera3D"
	cam.fov = 60.0
	add_child(cam)

	# Only add directional light if no lights exist in the scene already.
	var has_lights := false
	for child in get_children():
		if child is Light3D or (child.get_child_count() > 0 and child.get_child(0) is Light3D):
			has_lights = true
			break
	if not has_lights:
		var light := DirectionalLight3D.new()
		light.name = "Sun"
		light.rotation_degrees = Vector3(-45, 30, 0)
		add_child(light)

	if not _capture_dir.is_empty():
		DirAccess.make_dir_recursive_absolute(_capture_dir + "/" + _scene_name)

	print("[SYNTHETIC] %s | camera=%s | %ds" % [ply_path, camera_mode, int(duration)])

func _apply_contract() -> void:
	var scene_id := BenchmarkSceneContract.scene_id_from_path(get_tree().current_scene.scene_file_path)
	var defaults := {
		"lane_id": scene_id,
		"duration_s": duration,
		"output_path": "",
		"capture_dir": "",
		"reference_dir": "",
		"camera_mode": camera_mode,
		"asset_path": "",
		"orchestrated": false,
	}
	var contract := BenchmarkSceneContract.resolve_contract(scene_id, defaults, _pending_contract)
	_orchestrated = bool(contract.get("orchestrated", false))
	duration = maxf(1.0, float(contract.get("duration_s", duration)))
	camera_mode = str(contract.get("camera_mode", camera_mode))
	_output_path = str(contract.get("output_path", ""))
	_capture_dir = str(contract.get("capture_dir", ""))
	_reference_dir = str(contract.get("reference_dir", ""))
	_lane_id = str(contract.get("lane_id", scene_id))
	ply_path = BenchmarkSceneContract.resolve_asset_path(
		scene_id,
		_lane_id,
		str(contract.get("asset_path", "")),
		ply_path,
	)

func _process(delta: float) -> void:
	_elapsed += delta
	_frame_count += 1
	if delta > 0:
		_fps_samples.append(1.0 / delta)
	var t = _elapsed / duration  # 0..1 progress.

	var cam = get_node_or_null("Camera3D")
	if not cam:
		return

	match camera_mode:
		"flythrough":
			_camera_flythrough(cam, t)
		"sweep":
			_camera_sweep(cam, t)
		_:
			_camera_orbit(cam, t)

	# Capture at progress milestones.
	if not _capture_dir.is_empty():
		for frac in _capture_fractions:
			if t >= frac and not _captured_at.has(frac):
				_captured_at[frac] = true
				_do_capture(frac)

	if _elapsed > duration:
		# Final capture.
		if not _capture_dir.is_empty() and not _captured_at.has(1.0):
			_captured_at[1.0] = true
			_do_capture(1.0)

		_finish()

func _do_capture(progress: float) -> void:
	var viewport := get_viewport()
	if not viewport:
		return
	var image := BenchmarkVisualMetrics.capture_viewport(viewport)
	if image == null or image.is_empty():
		return

	var tag := "p%03d" % int(progress * 100)
	var filename := "%s/%s/%s_capture_%s.png" % [_capture_dir, _scene_name, _scene_name, tag]
	var err := BenchmarkVisualMetrics.save_png(image, filename)

	var capture_entry := {
		"progress": progress,
		"path": filename,
		"saved": err == OK,
	}

	# Compare against reference if available.
	if not _reference_dir.is_empty():
		var ref_path := "%s/%s/%s_capture_%s.png" % [_reference_dir, _scene_name, _scene_name, tag]
		var ref_image := BenchmarkVisualMetrics.load_image(ref_path)
		if ref_image != null and not ref_image.is_empty():
			capture_entry["ssim"] = BenchmarkVisualMetrics.calculate_ssim(image, ref_image)
			capture_entry["psnr"] = BenchmarkVisualMetrics.calculate_psnr(image, ref_image)
			capture_entry["reference_path"] = ref_path

	_captures.append(capture_entry)
	print("[SYNTHETIC] Captured %s (progress=%.0f%%)" % [filename, progress * 100])

func _finish() -> void:
	var avg_fps := 0.0
	if _elapsed > 0:
		avg_fps = _frame_count / _elapsed

	var p1_fps := 0.0
	if _fps_samples.size() > 0:
		var sorted := _fps_samples.duplicate()
		sorted.sort()
		p1_fps = sorted[int(sorted.size() * 0.01)]

	print("[SYNTHETIC] Done. %d frames / %.1fs = %.1f FPS (P1=%.1f)" % [_frame_count, _elapsed, avg_fps, p1_fps])

	# Compute visual summary.
	var ssim_values: Array[float] = []
	var psnr_values: Array[float] = []
	for c in _captures:
		if c.has("ssim"):
			ssim_values.append(c["ssim"])
		if c.has("psnr"):
			psnr_values.append(c["psnr"])

	var report := {
		"name": "GodotGS Synthetic Benchmark",
		"lane_id": _lane_id,
		"scene": _scene_name,
		"ply_path": ply_path,
		"camera_mode": camera_mode,
		"duration_s": _elapsed,
		"frame_count": _frame_count,
		"avg_fps": avg_fps,
		"p1_fps": p1_fps,
		"captures": _captures,
		"capture_count": _captures.size(),
	}

	if ssim_values.size() > 0:
		report["ssim_min"] = ssim_values.min()
		report["ssim_max"] = ssim_values.max()
		report["ssim_mean"] = _array_mean(ssim_values)
	if psnr_values.size() > 0:
		report["psnr_min"] = psnr_values.min()
		report["psnr_max"] = psnr_values.max()
		report["psnr_mean"] = _array_mean(psnr_values)

	# Write JSON report.
	if not _output_path.is_empty():
		var json_str := JSON.stringify(report, "  ")
		var f := FileAccess.open(_output_path, FileAccess.WRITE)
		if f:
			f.store_string(json_str)
			f.close()
			print("[SYNTHETIC] Report saved to %s" % _output_path)
	if _orchestrated:
		emit_signal("benchmark_scene_finished", {
			"success": true,
			"lane_id": _lane_id,
			"report_path": _output_path,
			"report": report,
		})
		queue_free()
		return
	get_tree().quit(0)

func _finish_with_error(message: String) -> void:
	if _orchestrated:
		emit_signal("benchmark_scene_finished", {
			"success": false,
			"lane_id": _lane_id,
			"report_path": _output_path,
			"error": message,
		})
		queue_free()
		return
	get_tree().quit(1)

func _array_mean(arr: Array[float]) -> float:
	var total := 0.0
	for v in arr:
		total += v
	return total / max(arr.size(), 1)

func _camera_orbit(cam: Camera3D, _t: float) -> void:
	var angle = _elapsed * orbit_speed
	cam.position = Vector3(sin(angle) * orbit_radius, orbit_height, cos(angle) * orbit_radius)
	cam.look_at(look_at_offset, Vector3.UP)

func _camera_flythrough(cam: Camera3D, t: float) -> void:
	# Fly from start to end with gentle spiral.
	var progress = clamp(t * fly_speed_scale, 0.0, 1.0)
	var pos = fly_start.lerp(fly_end, progress)
	# Add gentle spiral offset for visual interest.
	var spiral_r = 0.5 * (1.0 - abs(progress - 0.5) * 2.0)  # Widest at middle.
	var spiral_angle = progress * 6.0 * PI
	pos.x += cos(spiral_angle) * spiral_r
	pos.y += sin(spiral_angle) * spiral_r
	cam.position = pos
	# Look ahead along the tunnel.
	var ahead_t = min(progress + fly_look_ahead / max(fly_start.distance_to(fly_end), 0.01), 0.999)
	var look_target = fly_start.lerp(fly_end, ahead_t)
	if pos.distance_to(look_target) > 0.01:
		cam.look_at(look_target, Vector3.UP)

func _camera_sweep(cam: Camera3D, t: float) -> void:
	# Diagonal sweep across a flat grid.
	var progress = clamp(t, 0.0, 1.0)
	var sweep_angle = progress * PI * 0.4 - PI * 0.2  # -36° to +36°.
	var x = sin(sweep_angle) * sweep_extent
	var z = cos(sweep_angle) * sweep_extent * 0.5
	cam.position = Vector3(x, sweep_height, z)
	cam.look_at(Vector3(x * 0.3, 0, z * 0.3), Vector3.UP)
