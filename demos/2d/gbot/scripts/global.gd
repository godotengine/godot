extends Node

var viewport_scale = 0.0

func _ready():
	var viewport = get_node("/root/").get_children()[1].get_viewport_rect().size
	viewport_scale =  780 / viewport.y
	