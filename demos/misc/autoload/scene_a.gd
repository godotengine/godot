
extends Panel

# Member variables here, example:
# var a=2
# var b="textvar"


func _ready():
	# Initalization here
	pass


func _on_goto_scene_pressed():
	get_node("/root/global").goto_scene("res://scene_b.scn")
	pass # Replace with function body
