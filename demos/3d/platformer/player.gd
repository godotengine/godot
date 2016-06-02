
extends RigidBody

# Member variables
const ANIM_FLOOR = 0
const ANIM_AIR_UP = 1
const ANIM_AIR_DOWN = 2

const SHOOT_TIME = 1.5
const SHOOT_SCALE = 2

const CHAR_SCALE = Vector3(0.3, 0.3, 0.3)

var facing_dir = Vector3(1, 0, 0)
var movement_dir = Vector3()

var jumping = false

var turn_speed = 40
var keep_jump_inertia = true
var air_idle_deaccel = false
var accel = 19.0
var deaccel = 14.0
var sharp_turn_threshold = 140

var max_speed = 3.1
var on_floor = false

var prev_shoot = false

var last_floor_velocity = Vector3()

var shoot_blend = 0


func adjust_facing(p_facing, p_target, p_step, p_adjust_rate, current_gn):
	var n = p_target # Normal
	var t = n.cross(current_gn).normalized()
	
	var x = n.dot(p_facing)
	var y = t.dot(p_facing)
	
	var ang = atan2(y,x)
	
	if (abs(ang) < 0.001): # Too small
		return p_facing
	
	var s = sign(ang)
	ang = ang*s
	var turn = ang*p_adjust_rate*p_step
	var a
	if (ang < turn):
		a = ang
	else:
		a = turn
	ang = (ang - a)*s
	
	return (n*cos(ang) + t*sin(ang))*p_facing.length()


func _integrate_forces(state):
	var lv = state.get_linear_velocity() # Linear velocity
	var g = state.get_total_gravity()
	var delta = state.get_step()
#	var d = 1.0 - delta*state.get_total_density()
#	if (d < 0):
#		d = 0
	lv += g*delta # Apply gravity
	
	var anim = ANIM_FLOOR
	
	var up = -g.normalized() # (up is against gravity)
	var vv = up.dot(lv) # Vertical velocity
	var hv = lv - up*vv # Horizontal velocity
	
	var hdir = hv.normalized() # Horizontal direction
	var hspeed = hv.length() # Horizontal speed
	
	var floor_velocity
	var onfloor = false
	
	if (state.get_contact_count() == 0):
		floor_velocity = last_floor_velocity
	else:
		for i in range(state.get_contact_count()):
			if (state.get_contact_local_shape(i) != 1):
				continue
			
			onfloor = true
			floor_velocity = state.get_contact_collider_velocity_at_pos(i)
			break
	
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
	
	var jump_attempt = Input.is_action_pressed("jump")
	var shoot_attempt = Input.is_action_pressed("shoot")
	
	var target_dir = (dir - up*dir.dot(up)).normalized()
	
	if (onfloor):
		var sharp_turn = hspeed > 0.1 and rad2deg(acos(target_dir.dot(hdir))) > sharp_turn_threshold
		
		if (dir.length() > 0.1 and !sharp_turn):
			if (hspeed > 0.001):
				#linear_dir = linear_h_velocity/linear_vel
				#if (linear_vel > brake_velocity_limit and linear_dir.dot(ctarget_dir) < -cos(Math::deg2rad(brake_angular_limit)))
				#	brake = true
				#else
				hdir = adjust_facing(hdir, target_dir, delta, 1.0/hspeed*turn_speed, up)
				facing_dir = hdir
			else:
				hdir = target_dir
			
			if (hspeed < max_speed):
				hspeed += accel*delta
		else:
			hspeed -= deaccel*delta
			if (hspeed < 0):
				hspeed = 0
		
		hv = hdir*hspeed
		
		var mesh_xform = get_node("Armature").get_transform()
		var facing_mesh = -mesh_xform.basis[0].normalized()
		facing_mesh = (facing_mesh - up*facing_mesh.dot(up)).normalized()
		facing_mesh = adjust_facing(facing_mesh, target_dir, delta, 1.0/hspeed*turn_speed, up)
		var m3 = Matrix3(-facing_mesh, up, -facing_mesh.cross(up).normalized()).scaled(CHAR_SCALE)
		
		get_node("Armature").set_transform(Transform(m3, mesh_xform.origin))
		
		if (not jumping and jump_attempt):
			vv = 7.0
			jumping = true
			get_node("sfx").play("jump")
	else:
		if (vv > 0):
			anim = ANIM_AIR_UP
		else:
			anim = ANIM_AIR_DOWN
		
		var hs
		if (dir.length() > 0.1):
			hv += target_dir*(accel*0.2)*delta
			if (hv.length() > max_speed):
				hv = hv.normalized()*max_speed
		else:
			if (air_idle_deaccel):
				hspeed = hspeed - (deaccel*0.2)*delta
				if (hspeed < 0):
					hspeed = 0
				
				hv = hdir*hspeed
	
	if (jumping and vv < 0):
		jumping = false
	
	lv = hv + up*vv
	
	if (onfloor):
		movement_dir = lv
		#lv += floor_velocity
		last_floor_velocity = floor_velocity
	else:
		if (on_floor):
			#if (keep_jump_inertia):
			#	lv += last_floor_velocity
			pass
		
		last_floor_velocity = Vector3()
		movement_dir = lv
	
	on_floor = onfloor
	
	state.set_linear_velocity(lv)
	
	if (shoot_blend > 0):
		shoot_blend -= delta*SHOOT_SCALE
		if (shoot_blend < 0):
			shoot_blend = 0
	
	if (shoot_attempt and not prev_shoot):
		shoot_blend = SHOOT_TIME
		var bullet = preload("res://bullet.scn").instance()
		bullet.set_transform(get_node("Armature/bullet").get_global_transform().orthonormalized())
		get_parent().add_child(bullet)
		bullet.set_linear_velocity(get_node("Armature/bullet").get_global_transform().basis[2].normalized()*20)
		PS.body_add_collision_exception(bullet.get_rid(), get_rid()) # Add it to bullet
		get_node("sfx").play("shoot")
	
	prev_shoot = shoot_attempt
	
	if (onfloor):
		get_node("AnimationTreePlayer").blend2_node_set_amount("walk", hspeed/max_speed)
	
	get_node("AnimationTreePlayer").transition_node_set_current("state", anim)
	get_node("AnimationTreePlayer").blend2_node_set_amount("gun", min(shoot_blend, 1.0))
#	state.set_angular_velocity(Vector3())


func _ready():
	get_node("AnimationTreePlayer").set_active(true)
