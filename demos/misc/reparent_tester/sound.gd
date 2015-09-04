
extends SamplePlayer2D

func _ready():
	set_process_input(true)

func _input(event):
	if(event.type == InputEvent.MOUSE_BUTTON):
		if(event.button_index == 1 && event.is_pressed()):
			var pos = event.pos
			if(pos.y > 50):
				play("sound")
	



