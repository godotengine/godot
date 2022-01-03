# meta-description: Classic movement for gravity games (FPS, TPS, ...)

extends _BASE_

const SPEED: float = 5.0
const JUMP_FORCE: float = 4.5

# Get the gravity from the project settings to be synced with RigidDynamicBody nodes.
var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")


func _physics_process(delta: float) -> void:
	# Add the gravity.
	if not is_on_floor():
		motion_velocity.y -= gravity * delta

	# Handle Jump.
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		motion_velocity.y = JUMP_FORCE

	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	var input_dir := Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
	var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if direction:
		motion_velocity.x = direction.x * SPEED
		motion_velocity.z = direction.z * SPEED
	else:
		motion_velocity.x = move_toward(motion_velocity.x, 0, SPEED)
		motion_velocity.z = move_toward(motion_velocity.z, 0, SPEED)

	move_and_slide()
