extends Node3D

## Renders a single synthetic splat scene with configurable camera path.
## Camera modes: "orbit" (default), "flythrough" (tunnel), "sweep" (flat grid).

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

# Wind (applied to the splat node).
@export var enable_wind: bool = false
@export var wind_strength: float = 0.5
@export var wind_direction: Vector3 = Vector3(1, 0, 0.3)

# Sweep params (overhead grid).
@export var sweep_height: float = 8.0
@export var sweep_extent: float = 6.0

var _elapsed := 0.0
var _frame_count := 0

func _ready() -> void:
	Engine.max_fps = 0
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

	for arg in OS.get_cmdline_args():
		if arg.begins_with("--ply-path="):
			ply_path = arg.substr(len("--ply-path="))
		elif arg.begins_with("--duration="):
			duration = float(arg.substr(len("--duration=")))
		elif arg.begins_with("--camera-mode="):
			camera_mode = arg.substr(len("--camera-mode="))

	if ply_path.is_empty():
		push_error("[SYNTHETIC] No ply_path set")
		get_tree().quit()
		return

	var splat_node = GaussianSplatNode3D.new()
	splat_node.name = "SplatNode"
	splat_node.set_ply_file_path(ply_path)
	splat_node.set_auto_load(true)
	add_child(splat_node)

	if enable_wind:
		splat_node.set("wind/enabled", true)
		splat_node.set("wind/strength", wind_strength)
		splat_node.set("wind/direction", wind_direction)

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

	print("[SYNTHETIC] %s | camera=%s | %ds" % [ply_path, camera_mode, int(duration)])

func _process(delta: float) -> void:
	_elapsed += delta
	_frame_count += 1
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

	if _elapsed > duration:
		var avg_fps = _frame_count / _elapsed
		print("[SYNTHETIC] Done. %d frames / %.1fs = %.1f FPS" % [_frame_count, _elapsed, avg_fps])
		get_tree().quit()

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
