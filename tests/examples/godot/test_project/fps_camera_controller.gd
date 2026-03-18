extends Camera3D

var speed := 10.0
var fast_speed := 30.0
var mouse_sensitivity := 0.002

# Jacobian diagnostic toggle states (synced from renderer)
var _jacobian_bypass_depth := false
var _jacobian_bypass_clamp := false
var _jacobian_invert_sign := false
var _max_conic_aspect := 10.0  # Current aspect ratio clamp
var _log_last_ms: Dictionary = {}

## Captures mouse input for FPS camera control.
## Controls: WASD=move, E/Q=up/down, Shift=fast, ESC=quit
## Debug: 1/2/3=Jacobian toggles, ,/.=aspect clamp, 0=reset
func _ready() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

## Locates the first Gaussian renderer in the scene tree.
func _find_gaussian_renderer():
	# Find any GaussianSplat3D node and get its renderer
	var root = get_tree().root
	return _find_renderer_recursive(root)

## Recursively searches for a node exposing get_renderer.
func _find_renderer_recursive(node: Node):
	if node.has_method("get_renderer"):
		var renderer = node.get_renderer()
		if renderer != null:
			return renderer
	for child in node.get_children():
		var result = _find_renderer_recursive(child)
		if result != null:
			return result
	return null

## Handles mouse look, exit, and Jacobian diagnostic hotkeys.
## @param event: Input event dispatched by the scene tree.
func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * mouse_sensitivity)
		rotate_object_local(Vector3.RIGHT, -event.relative.y * mouse_sensitivity)
		rotation.x = clamp(rotation.x, deg_to_rad(-89), deg_to_rad(89))

	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_ESCAPE:
			get_tree().quit()

		# Jacobian diagnostic toggles (1, 2, 3 keys)
		if event.keycode == KEY_1:
			_toggle_jacobian_bypass_depth()
		elif event.keycode == KEY_2:
			_toggle_jacobian_bypass_clamp()
		elif event.keycode == KEY_3:
			_toggle_jacobian_invert_sign()
		# Aspect ratio clamp controls (comma, period, 0)
		elif event.keycode == KEY_COMMA:
			_adjust_aspect_clamp(-1.0)
		elif event.keycode == KEY_PERIOD:
			_adjust_aspect_clamp(1.0)
		elif event.keycode == KEY_0:
			_set_aspect_clamp(10.0)

## Toggles the radius depth-floor bypass in the renderer.
func _toggle_jacobian_bypass_depth() -> void:
	var renderer = _find_gaussian_renderer()
	if renderer == null:
		_gs_log_info("[JACOBIAN] No renderer found", "jacobian_no_renderer_depth")
		return
	_jacobian_bypass_depth = !_jacobian_bypass_depth
	renderer.set_jacobian_bypass_radius_depth_floor(_jacobian_bypass_depth)
	_gs_log_info("[JACOBIAN] bypass_radius_depth_floor = %s" % ("ON" if _jacobian_bypass_depth else "OFF"),
			"jacobian_depth_toggle")

## Toggles the Jacobian column clamp bypass in the renderer.
func _toggle_jacobian_bypass_clamp() -> void:
	var renderer = _find_gaussian_renderer()
	if renderer == null:
		_gs_log_info("[JACOBIAN] No renderer found", "jacobian_no_renderer_clamp")
		return
	_jacobian_bypass_clamp = !_jacobian_bypass_clamp
	renderer.set_jacobian_bypass_j_col2_clamp(_jacobian_bypass_clamp)
	_gs_log_info("[JACOBIAN] bypass_j_col2_clamp = %s" % ("ON" if _jacobian_bypass_clamp else "OFF"),
			"jacobian_clamp_toggle")

## Toggles the Jacobian sign inversion flag in the renderer.
func _toggle_jacobian_invert_sign() -> void:
	var renderer = _find_gaussian_renderer()
	if renderer == null:
		_gs_log_info("[JACOBIAN] No renderer found", "jacobian_no_renderer_sign")
		return
	_jacobian_invert_sign = !_jacobian_invert_sign
	renderer.set_jacobian_invert_j_col2_sign(_jacobian_invert_sign)
	_gs_log_info("[JACOBIAN] invert_j_col2_sign = %s" % ("ON" if _jacobian_invert_sign else "OFF"),
			"jacobian_sign_toggle")

## Adjusts the max conic aspect ratio clamp by the given delta.
## @param delta: Increment applied to the current clamp.
func _adjust_aspect_clamp(delta: float) -> void:
	var renderer = _find_gaussian_renderer()
	if renderer == null:
		_gs_log_info("[ASPECT] No renderer found", "aspect_no_renderer_adjust")
		return
	_max_conic_aspect = clamp(_max_conic_aspect + delta, 1.0, 20.0)
	renderer.set_max_conic_aspect(_max_conic_aspect)
	_gs_log_info("[ASPECT] max_conic_aspect = %.1f" % _max_conic_aspect, "aspect_adjust")

## Sets the max conic aspect ratio clamp to an explicit value.
## @param value: Clamp value to apply.
func _set_aspect_clamp(value: float) -> void:
	var renderer = _find_gaussian_renderer()
	if renderer == null:
		_gs_log_info("[ASPECT] No renderer found", "aspect_no_renderer_set")
		return
	_max_conic_aspect = value
	renderer.set_max_conic_aspect(_max_conic_aspect)
	_gs_log_info("[ASPECT] max_conic_aspect = %.1f (reset)" % _max_conic_aspect, "aspect_reset")

func _gs_debug_flag() -> String:
	if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_all_debug", false):
		return "enable_all_debug"
	if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_frame_logging", false):
		return "enable_frame_logging"
	if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_pipeline_trace", false):
		return "enable_pipeline_trace"
	if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_data_logging", false):
		return "enable_data_logging"
	return ""

func _gs_allow_log(key: String) -> bool:
	var rate_ms = ProjectSettings.get_setting("rendering/gaussian_splatting/logging/rate_limit_ms", 1000)
	if typeof(rate_ms) != TYPE_INT and typeof(rate_ms) != TYPE_FLOAT:
		rate_ms = 1000
	rate_ms = max(int(rate_ms), 0)
	if rate_ms <= 0:
		return true
	var now = Time.get_ticks_msec()
	var last = _log_last_ms.get(key, -1)
	if last >= 0 and now - last < rate_ms:
		return false
	_log_last_ms[key] = now
	return true

func _gs_log_info(message: String, key: String) -> void:
	var flag = _gs_debug_flag()
	if flag == "":
		return
	if not _gs_allow_log(key):
		return
	print("%s (debug: %s)" % [message, flag])

## Applies WASD-style movement each physics frame.
## @param delta: Physics frame delta in seconds.
func _physics_process(delta: float) -> void:
	var dir := Vector3.ZERO

	# Direct key checks
	if Input.is_key_pressed(KEY_W):
		dir.z -= 1.0
	if Input.is_key_pressed(KEY_S):
		dir.z += 1.0
	if Input.is_key_pressed(KEY_A):
		dir.x -= 1.0
	if Input.is_key_pressed(KEY_D):
		dir.x += 1.0
	if Input.is_key_pressed(KEY_E) or Input.is_key_pressed(KEY_SPACE):
		dir.y += 1.0
	if Input.is_key_pressed(KEY_Q):
		dir.y -= 1.0

	if dir.length_squared() > 0.001:
		# Transform direction by camera basis (local to world)
		var world_dir := transform.basis * dir
		world_dir = world_dir.normalized()
		var spd := fast_speed if Input.is_key_pressed(KEY_SHIFT) else speed
		global_position += world_dir * spd * delta
