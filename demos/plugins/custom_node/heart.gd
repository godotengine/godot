tool
extends Node2D


var heart = preload("res://addons/custom_node/heart.png")

func _draw():
	draw_texture(heart,-heart.get_size()/2)

func _get_item_rect():
	#override
	return Rect2(-heart.get_size()/2,heart.get_size())
