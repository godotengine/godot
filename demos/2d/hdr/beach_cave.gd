
extends Node2D

# Member variables
const CAVE_LIMIT = 1000


func _input(event):
	if (event.type == InputEvent.MOUSE_MOTION and event.button_mask&1):
		var rel_x = event.relative_x
		var cavepos = get_node("cave").get_pos()
		cavepos.x += rel_x
		if (cavepos.x < -CAVE_LIMIT):
			cavepos.x = -CAVE_LIMIT
		elif (cavepos.x > 0):
			cavepos.x = 0
		get_node("cave").set_pos(cavepos)


func _ready():
	# Initialization here
	set_process_input(true)
