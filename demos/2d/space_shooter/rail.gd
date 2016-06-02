
extends Node2D

# Member variables
const SPEED = 200
var offset = 0


func stop():
	set_fixed_process(false)


func _fixed_process(delta):
	offset += delta*SPEED
	set_pos(Vector2(offset, 0))


func _ready():
	set_fixed_process(true)
