
extends Area2D

#virtual from CollisionObject2D (also available as signal)
func _input_event(viewport, event, shape_idx):
	#convert event to local coordinates
	if (event.type==InputEvent.MOUSE_MOTION):
		event = make_input_local( event )
		get_node("label").set_text(str(event.pos))
		
#virtual from CollisionObject2D (also available as signal)
func _mouse_exit():
		get_node("label").set_text("")
		


