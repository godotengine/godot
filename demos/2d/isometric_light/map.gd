
extends Node2D

# member variables here, example:
# var a=2
# var b="textvar"

func _ready():
	# Initialization here
	pass




func _on_prince_area_body_enter( body ):
	if (body.get_name()=="cubio"):
		get_node("message").show()
	pass # replace with function body
