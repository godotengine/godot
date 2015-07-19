
extends Area2D

# member variables here, example:
# var a=2
# var b="textvar"

const SPEED = -800

func _process(delta):
	translate(Vector2(delta*SPEED,0))

func _ready():
	# Initialization here
	set_process(true)


var hit=false

func is_enemy():
	return true

func _hit_something():
	if (hit):
		return
	hit=true
	set_process(false)
	get_node("anim").play("splash")

func _on_visibility_exit_screen():
	queue_free()
	
