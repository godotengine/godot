
extends Panel

# member variables here, example:
# var a=2
# var b="textvar"

func _ready():
	# Initialization here
	pass




func _on_back_pressed():
	var s = load("res://main.scn")
	var si = s.instance()
	get_parent().add_child(si)
	queue_free()
	pass # replace with function body
