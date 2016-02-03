
extends Node2D

# member variables here, example:
# var a=2
# var b="textvar"
var firstTime = true
var startPos
func _ready():
	if(firstTime):
		startPos = get_parent().get_pos()
		self.set_fixed_process(true)
		firstTime = false

func _fixed_process(delta):
	get_parent().set_pos(get_parent().get_pos() + Vector2(0.1,0.1))
