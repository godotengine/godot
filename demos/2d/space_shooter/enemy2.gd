
extends Area2D

# Member variables
const SPEED = -220
const SHOOT_INTERVAL = 1

var shoot_timeout = 0
var destroyed=false


func _process(delta):
	translate(Vector2(SPEED*delta, 0))
	shoot_timeout -= delta
	
	if (shoot_timeout < 0):
		shoot_timeout = SHOOT_INTERVAL
		
		# Instance a shot
		var shot = preload("res://enemy_shot.scn").instance()
		# Set pos as "shoot_from" Position2D node
		shot.set_pos(get_node("shoot_from").get_global_pos())
		# Add it to parent, so it has world coordinates
		get_parent().add_child(shot)


func is_enemy():
	return not destroyed


func destroy():
	if (destroyed):
		return
	destroyed = true
	get_node("anim").play("explode")
	set_process(false)
	get_node("sfx").play("sound_explode")
	# Accumulate points
	get_node("/root/game_state").points += 10


func _ready():
	# Initialization here
	set_fixed_process(true)


func _on_visibility_enter_screen():
	set_process(true)


func _on_visibility_exit_screen():
	queue_free()
