
extends Node2D


func _on_prince_area_body_enter(body):
	if (body.get_name() == "cubio"):
		get_node("message").show()
