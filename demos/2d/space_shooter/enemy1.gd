
extends Area2D

# Member variables
const SPEED = -200

var destroyed=false


func _fixed_process(delta):
	get_parent().translate(Vector2(SPEED*delta, 0))


func is_enemy():
	return not destroyed


func destroy():
	if (destroyed):
		return
	destroyed = true
	get_node("anim").play("explode")
	set_fixed_process(false)
	get_node("sfx").play("sound_explode")
	# Accumulate points
	get_node("/root/game_state").points += 5


func _on_visibility_enter_screen():
	set_fixed_process(true)
	get_node("anim").play("zigzag")
	get_node("anim").seek(randf()*2.0) # Make it start from any pos


func _on_visibility_exit_screen():
	queue_free()
