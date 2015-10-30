
extends RigidBody

# member variables here, example:
# var a=2
# var b="textvar"

const STATE_WALKING = 0
const STATE_DYING = 1

                                     
var prev_advance = false
var deaccel = 20.0
var accel = 5
var max_speed = 2
var rot_dir = 4
var rot_speed = 1

var dying=false

func _integrate_forces(state):


	
	var delta = state.get_step()
	var lv = state.get_linear_velocity()
	var g = state.get_total_gravity()

	lv += g * delta #apply gravity
	var up = -g.normalized()

	if (dying):
		state.set_linear_velocity(lv)
		return
		
	for i in range(state.get_contact_count()):
		var cc = state.get_contact_collider_object(i)
		var dp = state.get_contact_local_normal(i)
	
		if (cc):
			if (cc extends preload("res://bullet.gd") and not cc.disabled):
				set_mode(MODE_RIGID)
				dying=true
				#lv=s.get_contact_local_normal(i)*400  
				state.set_angular_velocity( -dp.cross(up).normalized() *33.0)
				get_node("AnimationPlayer").play("impact")
				get_node("AnimationPlayer").queue("explode")
				set_friction(1)
				cc.disabled=true
				get_node("sound").play("hit")
				return

	
 		
	

	var col_floor = get_node("Armature/ray_floor").is_colliding()
	var col_wall = get_node("Armature/ray_wall").is_colliding()
	
	var advance = not col_wall and col_floor
	
	var dir = get_node("Armature").get_transform().basis[2].normalized()
	
	
	var deaccel_dir = dir
	
	if (advance):
		if (dir.dot(lv) < max_speed):
			lv+=dir * accel * delta
		deaccel_dir = dir.cross(g).normalized()
	else:
		if (prev_advance):
			rot_dir = 1 #randf() * 2.0 -1.0
		
		dir = Matrix3(up,rot_dir * rot_speed * delta).xform(dir)
		get_node("Armature").set_transform( Transform().looking_at(-dir,up) )
		
		
	var dspeed = deaccel_dir.dot(lv)
	dspeed -= deaccel * delta
	if (dspeed<0):
		dspeed=0
	
	lv = lv - deaccel_dir*deaccel_dir.dot(lv) + deaccel_dir*dspeed
			
	
	state.set_linear_velocity(lv)
	prev_advance=advance

func _ready():
	# Initalization here
	pass

func _die():
	queue_free()
