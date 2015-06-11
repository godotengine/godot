
extends Area2D

# member variables here, example:
# var a=2
# var b="textvar"

const SPEED=-200

func _process(delta):
	get_parent().translate(Vector2(SPEED*delta,0))

	
var destroyed=false

func is_enemy():
	return not destroyed


func destroy():
	if (destroyed):
		return	
	destroyed=true
	get_node("anim").play("explode")
	set_process(false)	
	get_node("sfx").play("sound_explode")
	#accum points
	get_node("/root/game_state").points+=5

func _on_visibility_enter_screen():
	set_process(true)
	get_node("anim").play("zigzag")	
	get_node("anim").seek(randf()*2.0) #make it start from any pos

func _on_visibility_exit_screen():
	queue_free()
	
