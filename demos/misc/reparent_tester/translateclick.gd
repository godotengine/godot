
extends Node2D

var firstTime = true
var startPos
var posState = 0
func _ready():
	if(firstTime):
		startPos = get_parent().get_pos()
		set_process_input(true)
		firstTime = false

func _fixed_process(delta):
	get_parent().set_pos(get_parent().get_pos() + Vector2(0.1,0.1))
	
func _input(event):
	if(event.type == InputEvent.MOUSE_BUTTON):
		if(event.button_index == 1 && event.is_pressed()):
			var pos = event.pos
			if(pos.y > 50):
				if(posState == 0):
					get_parent().set_pos(get_parent().get_pos() + Vector2(10,10))
					posState = 1
				else:
					get_parent().set_pos(get_parent().get_pos() - Vector2(10,10))
					posState = 0
