
extends Area

# member variables
var taken = false


func _on_coin_body_enter(body):
	if (not taken and body extends preload("res://player.gd")):
		get_node("anim").play("take")
		taken=true
