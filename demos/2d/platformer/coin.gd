
extends Area2D

# Member variables
var taken = false


func _on_body_enter( body ):
	if (not taken and body extends preload("res://player.gd")):
		get_node("anim").play("taken")
		taken = true


func _on_coin_area_enter(area):
	pass # replace with function body


func _on_coin_area_enter_shape(area_id, area, area_shape, area_shape):
	pass # replace with function body
