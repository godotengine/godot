
extends KinematicBody2D

# Member variables
const MAX_SPEED = 300.0
const IDLE_SPEED = 10.0
const ACCEL = 5.0
const VSCALE = 0.5
const SHOOT_INTERVAL = 0.3

var speed = Vector2()
var current_anim = ""
var current_mirror = false

var shoot_countdown = 0


func _input(event):
	if (event.type == InputEvent.MOUSE_BUTTON and event.button_index == 1 and event.pressed and shoot_countdown <= 0):
		var pos = get_canvas_transform().affine_inverse()*event.pos
		var dir = (pos - get_global_pos()).normalized()
		var bullet = preload("res://shoot.scn").instance()
		bullet.advance_dir = dir
		bullet.set_pos(get_global_pos() + dir*60)
		get_parent().add_child(bullet)
		shoot_countdown = SHOOT_INTERVAL


func _fixed_process(delta):
	shoot_countdown -= delta
	var dir = Vector2()
	if (Input.is_action_pressed("up")):
		dir += Vector2(0, -1)
	if (Input.is_action_pressed("down")):
		dir += Vector2(0, 1)
	if (Input.is_action_pressed("left")):
		dir += Vector2(-1, 0)
	if (Input.is_action_pressed("right")):
		dir += Vector2(1, 0)
	
	if (dir != Vector2()):
		dir = dir.normalized()
	speed = speed.linear_interpolate(dir*MAX_SPEED, delta*ACCEL)
	var motion = speed*delta
	motion.y *= VSCALE
	motion = move(motion)
	
	if (is_colliding()):
		var n = get_collision_normal()
		motion = n.slide(motion)
		move(motion)

	var next_anim = ""
	var next_mirror = false
	
	if (dir == Vector2() and speed.length() < IDLE_SPEED):
		next_anim = "idle"
		next_mirror = false
	elif (speed.length() > IDLE_SPEED*0.1):
		var angle = atan2(abs(speed.x), speed.y)
		
		next_mirror = speed.x > 0
		if (angle < PI/8):
			next_anim = "bottom"
			next_mirror = false
		elif (angle < PI/4 + PI/8):
			next_anim = "bottom_left"
		elif (angle < PI*2/4 + PI/8):
			next_anim = "left"
		elif (angle < PI*3/4 + PI/8):
			next_anim = "top_left"
		else:
			next_anim = "top"
			next_mirror = false
	
	if (next_anim != current_anim or next_mirror != current_mirror):
		get_node("frames").set_flip_h(next_mirror)
		get_node("anim").play(next_anim)
		current_anim = next_anim
		current_mirror = next_mirror


func _ready():
	# Initialization here
	set_fixed_process(true)
	set_process_input(true)
