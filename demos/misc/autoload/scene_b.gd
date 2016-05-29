extends Panel


func _on_goto_scene_pressed():
	get_node("/root/global").goto_scene("res://scene_a.scn")
