
extends RigidBody2D

# member variables here, example:
# var a=2
# var b="textvar"

var disabled=false

func disable():
	if (disabled):
		return
	get_node("anim").play("shutdown")
	disabled=true

func _ready():
	# Initalization here
	pass


