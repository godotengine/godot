
extends Node2D


const SPEED=200
# member variables here, example:
# var a=2
# var b="textvar"

func stop():
	set_process(false)

var offset=0


func _process(delta):
	offset+=delta*SPEED
	set_pos(Vector2(offset,0))

func _ready():
	set_process(true)
	# Initialization here
	


