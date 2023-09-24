# meta-description: Godot character controller for gravity games (FPS, TPS, ...)

extends _BASE_


@export var walk_acceleration := 15.0
@export var walk_max_acceleration := 300.0
@export var run_acceleration := walk_acceleration * 1.5
@export var run_max_acceleration := walk_max_acceleration * 1.5
@export var friction := 200.0
@export var jump_velocity := 5.0
@export_group("Camera")
@export var mouse_sensitivity := 0.005
@export var camera_max_vertical_angle := 30
@export var camera_min_vertical_angle := -10

# Get the global gravity from the project settings. RigidBody3D nodes also use this value.
var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")
var _current_acceleration := 0.0

# Nodes for camera management, replace these nodes by yours and remove the initialization in _ready.
var spring_arm := SpringArm3D.new()
var camera := Camera3D.new()


func _ready() -> void:
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	# Nodes for the camera, a SpringArm3D that has a Camera3D as a child (to be deleted when using your nodes).
	camera.current = true
	spring_arm.position = Vector3(0, 1, 0) # Camera position relative to the player origin.
	spring_arm.spring_length = 5 # Set to 0 to get an FPS-like camera.
	spring_arm.add_child(camera)
	add_child(spring_arm)


func _unhandled_input(event) -> void:
	# The mouse cursor is not visible in the game, you can toggle the behavior with the escape key.
	if event.is_action_pressed(&"ui_cancel"):
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED if Input.get_mouse_mode() != Input.MOUSE_MODE_CAPTURED else Input.MOUSE_MODE_VISIBLE)

	# Moves the camera according to the movement of the cursor.
	elif Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED and event is InputEventMouseMotion:
		rotate_y(-event.relative.x * mouse_sensitivity)
		spring_arm.rotate_x(-event.relative.y * mouse_sensitivity)
		spring_arm.rotation.x = clamp(spring_arm.rotation.x, deg_to_rad(camera_min_vertical_angle), deg_to_rad(camera_max_vertical_angle))


func _physics_process(delta: float) -> void:
	# As good practice, you should replace UI actions with custom gameplay actions.
	var running = Input.is_action_pressed(&"ui_focus_next")
	var acceleration := run_acceleration if running else walk_acceleration
	var max_acceleration: float = run_max_acceleration if running else walk_max_acceleration

	# Add the gravity.
	if not is_on_floor():
		velocity.y -= gravity * delta

	# Allow jumping only when on the floor.
	if Input.is_action_just_pressed(&"ui_accept") and is_on_floor():
		velocity.y = jump_velocity

	# Apply friction.
	_current_acceleration = move_toward(_current_acceleration, 0, friction * delta)
	velocity.x = move_toward(velocity.x, 0, friction * delta)
	velocity.z = move_toward(velocity.z, 0, friction * delta)

	# Get the input direction and handle the movement/deceleration.
	var input_dir := Input.get_vector(&"ui_left", &"ui_right", &"ui_up", &"ui_down")
	var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if direction:
		_current_acceleration += acceleration
		_current_acceleration = clamp(_current_acceleration, 0, max_acceleration)
		velocity.x = _current_acceleration * direction.x * delta
		velocity.z = _current_acceleration * direction.z * delta

	move_and_slide()
