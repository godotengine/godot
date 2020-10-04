extends KinematicBody

export var speed : float = 20
export var acceleration : float = 5
export var gravity : float = 0.98

var velocity : Vector3
var fall_velocity : float
var isJumping : bool = false
var isFlying : bool = false

func _physics_process(delta):
	move_player(delta)

func move_player(delta):
	var direction = Vector3(0,0,0)
	
	if Input.is_action_pressed("move_right"):
		direction = Vector3(1, 0, 0)
	
	if Input.is_action_pressed("move_left"):
		direction = Vector3(direction.x - 1, 0, 0)
	
	if Input.is_action_pressed("move_backward"):
		direction = Vector3(direction.x, 0, 1)
	
	if Input.is_action_pressed("move_forward"):
		direction = Vector3(direction.x, 0, direction.z - 1)
	
	direction = direction.normalized()
	velocity = velocity.linear_interpolate(direction * speed, acceleration * delta)
		
	if is_on_floor():
		fall_velocity = -0.01
	else:
		fall_velocity = fall_velocity - gravity
		
	if Input.is_action_pressed("jump") &&  !isJumping && !Input.is_action_pressed("Fly"):
		fall_velocity = 15
		isJumping = true
	
	if Input.is_action_pressed("Fly") && Input.is_action_pressed("jump"):
		fall_velocity = 15
		isFlying = true
	
	if is_on_floor() && isJumping:
		isJumping = false
	
	velocity.y = fall_velocity
	velocity = move_and_slide(velocity, Vector3.UP)
