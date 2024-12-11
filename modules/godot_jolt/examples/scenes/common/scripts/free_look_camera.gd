class_name FreeLookCamera3D extends Camera3D

@export_range(0.0, 1000.0, 0.1, "or_greater", "exp", "suffix:u/s")
var initial_speed := 10.0

@export_range(0.0, 100.0, 0.5, "or_greater")
var interpolation_speed := 15.0

@export_range(0.01, 10.0, 0.01, "or_greater", "exp")
var mouse_sensitivity := 1.0

@export_range(1.0, 2.0, 0.01, "or_greater", "exp")
var speed_step_factor := 1.2

@export_range(0.1, 1000.0, 0.1, "or_greater")
var pick_reach := 1000.0

@export_range(0.1, 100.0, 0.1, "or_greater", "exp")
var pick_frequency := 5.0

@export_range(1.0, 100.0, 0.1, "or_greater", "exp")
var pick_damping := 1.0

@export_range(2, 64, 1)
var pick_iterations := 64

var speed := initial_speed

var picking := false
var pick_depth := 0.0

var pick_anchor: AnimatableBody3D
var pick_joint: JoltGeneric6DOFJoint3D

@onready var target_position := position

func _enter_tree() -> void:
	pick_anchor = AnimatableBody3D.new()
	pick_anchor.collision_layer = 0
	pick_anchor.collision_mask = 0

	pick_joint = JoltGeneric6DOFJoint3D.new()
	pick_joint["angular_limit_x/enabled"] = false
	pick_joint["angular_limit_y/enabled"] = false
	pick_joint["angular_limit_z/enabled"] = false
	pick_joint["linear_limit_spring_x/enabled"] = true
	pick_joint["linear_limit_spring_y/enabled"] = true
	pick_joint["linear_limit_spring_z/enabled"] = true
	pick_joint["linear_limit_spring_x/frequency"] = pick_frequency
	pick_joint["linear_limit_spring_y/frequency"] = pick_frequency
	pick_joint["linear_limit_spring_z/frequency"] = pick_frequency
	pick_joint["linear_limit_spring_x/damping"] = pick_damping
	pick_joint["linear_limit_spring_y/damping"] = pick_damping
	pick_joint["linear_limit_spring_z/damping"] = pick_damping
	pick_joint["solver_velocity_iterations"] = pick_iterations
	pick_joint["solver_position_iterations"] = pick_iterations

func _exit_tree() -> void:
	pick_joint.queue_free()
	pick_anchor.queue_free()

func _input(event: InputEvent) -> void:
	if not current:
		return

	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				_start_picking(event.position)
			else:
				_stop_picking()
		if event.button_index == MOUSE_BUTTON_RIGHT:
			if event.pressed:
				Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
			else:
				Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			if event.pressed:
				_step_speed_up(event.factor)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			if event.pressed:
				_step_speed_down(event.factor)
	elif event is InputEventMouseMotion:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			_rotate_from_mouse(event.relative)

func _process(delta: float) -> void:
	if not current:
		return

	var step := speed * delta

	if Input.is_physical_key_pressed(KEY_SHIFT):
		step *= 4
	if Input.is_physical_key_pressed(KEY_ALT):
		step /= 4

	if Input.is_physical_key_pressed(KEY_W):
		target_position += -basis.z * step
	if Input.is_physical_key_pressed(KEY_S):
		target_position += basis.z * step
	if Input.is_physical_key_pressed(KEY_A):
		target_position += -basis.x * step
	if Input.is_physical_key_pressed(KEY_D):
		target_position += basis.x * step

	position = position.lerp(target_position, interpolation_speed * delta)

func _physics_process(_delta: float) -> void:
	_move_pick(get_viewport().get_mouse_position())

func _start_picking(mouse_position: Vector2) -> void:
	if picking:
		return

	var ray_from := project_ray_origin(mouse_position)
	var ray_dir := project_ray_normal(mouse_position)
	var ray_to := ray_from + ray_dir * pick_reach
	var ray_params := PhysicsRayQueryParameters3D.create(ray_from, ray_to)
	var ray_hit := get_world_3d().direct_space_state.intersect_ray(ray_params)

	if ray_hit and ray_hit.collider is RigidBody3D:
		var hit_pos := to_local(ray_hit.position)

		pick_anchor.position = hit_pos
		pick_joint.position = hit_pos

		add_child(pick_anchor)
		add_child(pick_joint)

		pick_joint.node_a = pick_anchor.get_path()
		pick_joint.node_b = ray_hit.collider.get_path()
		pick_depth = ray_from.distance_to(ray_hit.position)

		picking = true

func _move_pick(mouse_position: Vector2) -> void:
	if not picking:
		return

	var ray_from := project_ray_origin(mouse_position)
	var ray_dir := project_ray_normal(mouse_position)

	pick_anchor.global_position = ray_from + ray_dir * pick_depth

func _stop_picking() -> void:
	if not picking:
		return

	pick_joint.node_a = NodePath()
	pick_joint.node_b = NodePath()

	remove_child(pick_joint)
	remove_child(pick_anchor)

	picking = false

func _step_speed_up(factor: float = 1.0) -> void:
	_scale_speed(1.0 + (speed_step_factor - 1.0) * factor)

func _step_speed_down(factor: float = 1.0) -> void:
	_scale_speed(1.0 / (1.0 + (speed_step_factor - 1.0) * factor))

func _scale_speed(factor: float) -> void:
	var min_speed := near * 4
	var max_speed := far / 4

	if min_speed < max_speed:
		speed = clampf(speed * factor, min_speed, max_speed);
	else:
		speed = (min_speed + max_speed) / 2.0;

func _rotate_from_mouse(relative: Vector2) -> void:
	var delta := -relative * 0.0015 * mouse_sensitivity

	rotation = Vector3(
		clampf(rotation.x + delta.y, -PI / 2.0, PI / 2.0),
		wrapf(rotation.y + delta.x, 0.0, TAU),
		rotation.z
	)
