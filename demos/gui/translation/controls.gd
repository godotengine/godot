
extends Panel


func _on_back_pressed():
	var s = load("res://main.scn")
	var si = s.instance()
	get_parent().add_child(si)
	queue_free()
