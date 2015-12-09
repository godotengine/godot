
extends Panel

# member variables here, example:
# var a=2
# var b="textvar"


func _ready():
	# Initalization here
	pass


func _on_goto_scene_pressed():
	get_node("/root/global").goto_scene("res://scene_a.scn")
	pass # replace with function body
