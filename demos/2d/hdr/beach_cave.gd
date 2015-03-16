
extends Node2D

# member variables here, example:
# var a=2
# var b="textvar"
const CAVE_LIMIT=1000

func _input(ev):
	if (ev.type==InputEvent.MOUSE_MOTION and ev.button_mask&1):
		var rel_x = ev.relative_x
		var cavepos = get_node("cave").get_pos()
		cavepos.x+=rel_x
		if (cavepos.x<-CAVE_LIMIT):
			cavepos.x=-CAVE_LIMIT
		elif (cavepos.x>0):
			cavepos.x=0
		get_node("cave").set_pos(cavepos)
			

func _ready():
	set_process_input(true)
	# Initialization here
	pass


