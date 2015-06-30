
extends Area2D

# member variables here, example:
# var a=2
# var b="textvar"
const SPEED=-220
const SHOOT_INTERVAL=1
var shoot_timeout=0

func _process(delta):
	translate( Vector2(SPEED*delta,0) )
	shoot_timeout-=delta
	
	if (shoot_timeout<0):
	
		shoot_timeout=SHOOT_INTERVAL
		
		#instance a shot
		var shot = preload("res://enemy_shot.scn").instance()
		#set pos as "shoot_from" Position2D node
		shot.set_pos( get_node("shoot_from").get_global_pos() )
		#add it to parent, so it has world coordinates
		get_parent().add_child(shot)
		
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
	get_node("/root/game_state").points+=10

func _ready():
	set_fixed_process(true)
	# Initialization here
	pass




func _on_visibility_enter_screen():
	set_process(true)
	pass # replace with function body


func _on_visibility_exit_screen():
	queue_free()
	pass # replace with function body
