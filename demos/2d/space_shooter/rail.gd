
extends Node2D

# Member variables
const SPEED = 200
var offset = 0


func stop():
	set_process(false)


func _process(delta):
	offset += delta*SPEED
	set_pos(Vector2(offset, 0))


func _ready():
	# Initialization here
	set_process(true)
