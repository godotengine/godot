
extends Area2D

# member variables here, example:
# var a=2
# var b="textvar"

const SPEED=-200
const Y_RANDOM=10

var points=1


var speed_y=0.0

func _process(delta):

	translate( Vector2(SPEED,speed_y) * delta )

func _ready():
	# Initialization here	
	speed_y=rand_range(-Y_RANDOM,Y_RANDOM)
	pass

var destroyed=false

func destroy():
	if (destroyed):
		return	
	destroyed=true
	get_node("anim").play("explode")
	set_process(false)
	get_node("sfx").play("sound_explode")
	#accum points
	get_node("/root/game_state").points+=1
	
func is_enemy():
	return not destroyed	
	

func _on_visibility_enter_screen():
	set_process(true)
	#make it spin!
	get_node("anim").play("spin")
	

func _on_visibility_exit_screen():
	queue_free()
	pass # replace with function body
