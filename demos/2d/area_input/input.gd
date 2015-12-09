
extends Area2D


# Virtual from CollisionObject2D (also available as signal)
func _input_event(viewport, event, shape_idx):
	# Convert event to local coordinates
	if (event.type == InputEvent.MOUSE_MOTION):
		event = make_input_local(event)
		get_node("label").set_text(str(event.pos))


# Virtual from CollisionObject2D (also available as signal)
func _mouse_exit():
	get_node("label").set_text("")
