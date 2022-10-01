# meta-description: Classic movement for gravity games (platformer, ...)

extends _BASE_


const SPEED = 300.0
const JUMP_VELOCITY = -400.0

# Get the global gravity from the project settings. RigidBody2D nodes also use this value.
var gravity: float = ProjectSettings.get_setting("physics/2d/default_gravity")


func _physics_process(delta: float) -> void:
	# Add the gravity.
	velocity.y += gravity * delta

	# Allow jumping only when on the floor.
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	# Get the input direction and handle the movement.
	# As good practice, you should replace UI actions with custom input actions.
	var direction := Input.get_axis("ui_left", "ui_right")
	velocity.x = direction * SPEED

	move_and_slide()
