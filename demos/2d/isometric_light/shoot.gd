
extends KinematicBody2D

# Member variables
const ADVANCE_SPEED = 500.0

var advance_dir = Vector2(1, 0)
var hit = false


func _fixed_process(delta):
	if (hit):
		return
	move(advance_dir*delta*ADVANCE_SPEED)
	if (is_colliding()):
		get_node("anim").play("explode")
		hit = true


func _ready():
	# Initialization here
	set_fixed_process(true)
