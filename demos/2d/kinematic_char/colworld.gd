
extends Node2D


func _on_princess_body_enter(body):
	# The name of this editor-generated callback is unfortunate
	if (body.get_name() == "player"):
		get_node("youwin").show()
