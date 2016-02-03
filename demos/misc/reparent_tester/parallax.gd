
extends Node2D

# member variables here, example:
# var a=2
# var b="textvar"
var camera
func _ready():
	# Initialization here
	camera = get_parent().get_parent().get_parent().get_node("Camera2D")
	set_fixed_process(true)

func _fixed_process(delta):
	camera.set_pos(camera.get_pos() + Vector2(-0.05,-0.05))

