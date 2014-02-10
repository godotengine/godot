
extends Spatial


func _on_pause_pressed():
	get_node("pause_popup").set_exclusive(true)
	get_node("pause_popup").popup()
	get_scene().set_pause(true)


func _on_unpause_pressed():
	get_node("pause_popup").hide()
	get_scene().set_pause(false)
	
	
