extends GutTest

class DraggableButton:
	extends Button

	func _get_drag_data(_at_position):
		var preview = self.duplicate(0)
		preview.modulate.a = .25
		set_drag_preview(preview)
		return self

var _sender

func before_all():
	_sender = InputSender.new(Input)

func after_each():
	_sender.release_all()
	_sender.clear()


# This test tries to use exclusively InputSender to press, drag, and release the
# control.  Without button_mask being set, this test will fail.
func test_drag_using_input_sender():
	var drag_me = add_child_autofree(DraggableButton.new())
	drag_me.size = Vector2(100, 100)

	await (_sender
		.mouse_set_position(Vector2(64,64)) # Set position to middle of control
		.mouse_left_button_down().wait_frames(5) # Begin click
		.mouse_relative_motion(Vector2(128,0)) # Move to the right to begin drag
		.wait_frames(5)).idle

	assert_eq(get_viewport().gui_get_drag_data(), drag_me, 'button being dragged')

	await _sender.mouse_left_button_up().wait_frames(5).idle # Release the mouse

	assert_null(get_viewport().gui_get_drag_data(), "nothing is being dragged")


# This test sets the button_mask by making its own mouse motion events.  This
# passes and illustrates what the other test should be doing.
func test_drag_using_input_factory():
	var drag_me = add_child_autofree(DraggableButton.new())
	drag_me.size = Vector2(100, 100)

	await _sender.mouse_left_button_down(Vector2(64,64)).wait_frames(5) # Mouse down on control

	var event = InputFactory.mouse_motion(Vector2(196,64)) # Create a mouse motion event at the destination
	event.relative = Vector2i(128,0) # Set the relative motion
	event.button_mask = MOUSE_BUTTON_MASK_LEFT # Indicate we are holding left click

	await _sender.send_event(event).wait_frames(5).idle # Drag event

	assert_eq(get_viewport().gui_get_drag_data(), drag_me, 'button being dragged')

	await _sender.mouse_left_button_up(Vector2(196,64)).wait_frames(5).idle # Release the mouse

	assert_null(get_viewport().gui_get_drag_data(), "nothing is being dragged")