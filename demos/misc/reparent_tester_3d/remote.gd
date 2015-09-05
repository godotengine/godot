
extends Node2D

# member variables here, example:
# var a=2
# var b="textvar"
var remote

func _ready():
	remote = get_node("RemoteTransform2D")
	set_fixed_process(true)

func _fixed_process(delta):
	remote.set_pos(remote.get_pos() + Vector2(0.1,0.1))




