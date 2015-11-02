
extends RigidBody2D

# member variables here, example:
# var a=2
# var b="textvar"

var timeout=5

func _process(delta):
	timeout-=delta
	if (timeout<1):
		set_opacity(timeout)
	if (timeout<0):
		queue_free()
func _ready():
	set_process(true)
	# Initialization here
	pass


