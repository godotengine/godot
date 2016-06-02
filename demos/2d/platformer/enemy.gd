
extends RigidBody2D

# Member variables
const STATE_WALKING = 0
const STATE_DYING = 1

var state = STATE_WALKING

var direction = -1
var anim = ""

var rc_left = null
var rc_right = null
var WALK_SPEED = 50

var bullet_class = preload("res://bullet.gd")


func _die():
	queue_free()


func _pre_explode():
	# Stay there
	clear_shapes()
	set_mode(MODE_STATIC)
	get_node("sound").play("explode")


func _integrate_forces(s):
	var lv = s.get_linear_velocity()
	var new_anim = anim

	if (state == STATE_DYING):
		new_anim = "explode"
	elif (state == STATE_WALKING):
		new_anim = "walk"
		
		var wall_side = 0.0
		
		for i in range(s.get_contact_count()):
			var cc = s.get_contact_collider_object(i)
			var dp = s.get_contact_local_normal(i)
			
			if (cc):
				if (cc extends bullet_class and not cc.disabled):
					set_mode(MODE_RIGID)
					state = STATE_DYING
					#lv = s.get_contact_local_normal(i)*400
					s.set_angular_velocity(sign(dp.x)*33.0)
					set_friction(1)
					cc.disable()
					get_node("sound").play("hit")
					break
			
			if (dp.x > 0.9):
				wall_side = 1.0
			elif (dp.x < -0.9):
				wall_side = -1.0
		
		if (wall_side != 0 and wall_side != direction):
			direction = -direction
			get_node("sprite").set_scale(Vector2(-direction, 1))
		if (direction < 0 and not rc_left.is_colliding() and rc_right.is_colliding()):
			direction = -direction
			get_node("sprite").set_scale(Vector2(-direction, 1))
		elif (direction > 0 and not rc_right.is_colliding() and rc_left.is_colliding()):
			direction = -direction
			get_node("sprite").set_scale(Vector2(-direction, 1))
		
		lv.x = direction*WALK_SPEED
	
	if(anim != new_anim):
		anim = new_anim
		get_node("anim").play(anim)
	
	s.set_linear_velocity(lv)


func _ready():
	rc_left = get_node("raycast_left")
	rc_right = get_node("raycast_right")
