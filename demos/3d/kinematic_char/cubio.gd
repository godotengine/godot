
extends KinematicBody

# Member variables
var g = -9.8
var vel = Vector3()
const MAX_SPEED = 5
const JUMP_SPEED = 7
const ACCEL= 2
const DEACCEL= 4
const MAX_SLOPE_ANGLE = 30


func _fixed_process(delta):
	var dir = Vector3() # Where does the player intend to walk to
	var cam_xform = get_node("target/camera").get_global_transform()
	
	if (Input.is_action_pressed("move_forward")):
		dir += -cam_xform.basis[2]
	if (Input.is_action_pressed("move_backwards")):
		dir += cam_xform.basis[2]
	if (Input.is_action_pressed("move_left")):
		dir += -cam_xform.basis[0]
	if (Input.is_action_pressed("move_right")):
		dir += cam_xform.basis[0]
	
	dir.y = 0
	dir = dir.normalized()
	
	vel.y += delta*g
	
	var hvel = vel
	hvel.y = 0
	
	var target = dir*MAX_SPEED
	var accel
	if (dir.dot(hvel) > 0):
		accel = ACCEL
	else:
		accel = DEACCEL
	
	hvel = hvel.linear_interpolate(target, accel*delta)
	
	vel.x = hvel.x
	vel.z = hvel.z
	
	var motion = move(vel*delta)
	
	var on_floor = false
	var original_vel = vel
	var floor_velocity = Vector3()
	var attempts = 4
	
	while(is_colliding() and attempts):
		var n = get_collision_normal()
		
		if (rad2deg(acos(n.dot(Vector3(0, 1, 0)))) < MAX_SLOPE_ANGLE):
				# If angle to the "up" vectors is < angle tolerance,
				# char is on floor
				floor_velocity = get_collider_velocity()
				on_floor = true
			
		motion = n.slide(motion)
		vel = n.slide(vel)
		if (original_vel.dot(vel) > 0):
			# Do not allow to slide towads the opposite direction we were coming from
			motion=move(motion)
			if (motion.length() < 0.001):
				break
		attempts -= 1
	
	if (on_floor and floor_velocity != Vector3()):
		move(floor_velocity*delta)
	
	if (on_floor and Input.is_action_pressed("jump")):
		vel.y = JUMP_SPEED
	
	var crid = get_node("../elevator1").get_rid()


func _ready():
	set_fixed_process(true)


func _on_tcube_body_enter(body):
	get_node("../ty").show()
