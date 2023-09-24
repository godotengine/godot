# meta-description: Godot character controller for gravity games (platformer, ...)

extends _BASE_


@export var walk_max_speed := 500.0
@export var walk_acceleration := 1500.0
@export var run_max_speed := walk_max_speed * 1.5
@export var run_acceleration := walk_acceleration * 1.5
@export var friction := 1000.0
@export var slide_friction := 1000.0
@export var jump_velocity := -400.0

# Get the global gravity from the project settings. RigidBody2D nodes also use this value.
var gravity: float = ProjectSettings.get_setting("physics/2d/default_gravity")


func _physics_process(delta: float) -> void:
	# As good practice, you should replace UI actions with custom input actions.
	var running = Input.is_action_pressed(&"ui_focus_next")
	var acceleration := run_acceleration if running else walk_acceleration
	var max_speed := run_max_speed if running else walk_max_speed

	# Add the gravity.
	if not is_on_floor():
		velocity.y += gravity * delta

	# Allow jumping only when on the floor.
	if Input.is_action_just_pressed(&"ui_accept") and is_on_floor():
		velocity.y = jump_velocity

	# Apply friction.
	velocity.x = move_toward(velocity.x, 0, friction * delta)

	# Get the input direction and handle the movement.
	var direction := Input.get_axis(&"ui_left", &"ui_right")
	if direction:
		# If the direction is opposite to the x-velocity, extra friction is applied for more reactivity.
		if direction + sign(velocity.x) == 0:
			velocity.x += slide_friction * direction * delta
		velocity.x = clampf(velocity.x + acceleration * direction * delta, -max_speed, max_speed)

	move_and_slide()
