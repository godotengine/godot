
extends Area2D

# Member variables
const SPEED = 200

var screen_size
var prev_shooting = false
var killed = false


func _process(delta):
	var motion = Vector2()
	if Input.is_action_pressed("move_up"):
		motion += Vector2(0, -1)
	if Input.is_action_pressed("move_down"):
		motion += Vector2(0, 1)
	if Input.is_action_pressed("move_left"):
		motion += Vector2(-1, 0)
	if Input.is_action_pressed("move_right"):
		motion += Vector2(1, 0)
	var shooting = Input.is_action_pressed("shoot")
	
	var pos = get_pos()
	
	pos += motion*delta*SPEED
	if (pos.x < 0):
		pos.x = 0
	if (pos.x > screen_size.x):
		pos.x = screen_size.x
	if (pos.y < 0):
		pos.y = 0
	if (pos.y > screen_size.y):
		pos.y = screen_size.y
	
	set_pos(pos)
	
	if (shooting and not prev_shooting):
		# Just pressed
		var shot = preload("res://shot.scn").instance()
		# Use the Position2D as reference
		shot.set_pos(get_node("shootfrom").get_global_pos())
		# Put it two parents above, so it is not moved by us
		get_node("../..").add_child(shot)
		# Play sound
		get_node("sfx").play("shoot")
	
	prev_shooting = shooting
	
	# Update points counter
	get_node("../hud/score_points").set_text(str(get_node("/root/game_state").points))


func _ready():
	# Initialization here
	screen_size = get_viewport().get_rect().size
	set_process(true)


func _hit_something():
	if (killed):
		return
	killed = true
	get_node("anim").play("explode")
	get_node("sfx").play("sound_explode")
	get_node("../hud/game_over").show()
	get_node("/root/game_state").game_over()
	get_parent().stop()
	set_process(false)


func _on_ship_body_enter(body):
	_hit_something()


func _on_ship_area_enter(area):
	if (area.has_method("is_enemy") and area.is_enemy()):
		_hit_something()


func _on_back_to_menu_pressed():
	get_tree().change_scene("res://main_menu.scn")
