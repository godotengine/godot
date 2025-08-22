extends GutTest

class SuperButton:
	extends Button

	func p(s1='', s2='', s3='', s4='', s5='', s6=''):
		print(s1, s2, s3, s4, s5, s6)

	func pevent(txt, event):
		return
		print(txt, ':  ', event)
		# if(event is InputEventMouse):
		# 	print(txt, ':  ', event.position, event.global_position)
		# else:
		# 	print(txt, ':  ', event)

	func _gui_input(event):
		pevent('gui:      ', event)

	func _input(event):
		pevent('input:     ', event)

	func _unhandled_input(event):
		pevent('unhandled:  ', event)


class DraggableButton:
	extends SuperButton

	var _mouse_down = false

	func _gui_input(event):
		super._gui_input(event)
		if(event is InputEventMouseButton):
			_mouse_down = event.pressed
		elif(event is InputEventMouseMotion and _mouse_down):
			position += event.relative

func should_skip_script():
	return 'takes too long and these shouldnt even be here'


func _print_emitted_signals(thing):
	_signal_watcher.print_signal_summary(thing)
	return


func test_draw_mouse():
	var sender = InputSender.new(Input)
	sender.mouse_warp = false
	sender.draw_mouse = true
	var pos = Vector2(200, 200)
	sender\
		.mouse_left_button_down(pos)\
		.wait(1)\
		.mouse_left_button_up()\
		.mouse_right_button_down()\
		.wait(1)\
		.mouse_right_button_up()\
		.mouse_relative_motion(Vector2(10, 10)).wait(.5)\
		.mouse_relative_motion(Vector2(10, 10)).wait(.5)\
		.mouse_left_button_down().hold_for(.5)\
		.mouse_relative_motion(Vector2(10, 10)).wait(.5)\
		.mouse_relative_motion(Vector2(10, 10)).wait(.5)\
		.mouse_right_button_down().hold_for(.5)\
		.mouse_relative_motion(Vector2(10, 10)).wait(.5)\
		.mouse_left_button_down()\
		.mouse_right_button_down()\
		.wait(2)

	await sender.idle


func test_drag_something():
	var btn = DraggableButton.new()
	watch_signals(btn)
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child_autofree(btn)

	# works with Input and btn, btn does not fire signals, Input seems to be
	# having some trouble firigin the button up event.
	var sender = InputSender.new(Input)
	sender.set_auto_flush_input(true)
	sender.mouse_warp = false
	sender.draw_mouse = true

	sender.mouse_left_button_down(btn.position + Vector2(10, 10)).wait(.1)
	for i in range(10):
		await sender.mouse_relative_motion(Vector2(10, 10)).wait(.1).idle
		print('-- ', btn.position, ' --')

	assert_true(Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT), 'left button is down')
	await sender\
		.mouse_left_button_up()\
		.wait('1f')\
		.mouse_relative_motion(Vector2(1, 1))\
		.wait(.2).idle

	assert_false(Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT), 'left button is up')
	var after_first_drag_pos = btn.position
	# drag again after mouse up which shouldn't move
	for i in range(10):
		await sender.mouse_relative_motion(Vector2(10, 10)).wait(.1).idle
		print('-- ', btn.position, ' --')

	_print_emitted_signals(btn)

	assert_signal_emitted(btn, 'button_down')
	assert_signal_emitted(btn, 'button_up')
	assert_ne(btn.position, Vector2(50, 50), 'has moved')
	assert_false(btn._mouse_down, 'button mouse down')
	assert_eq(btn.position, after_first_drag_pos, 'does not move after releasing button')



#     50 ->|         |<- 150
func test_clicking_things_with_input_as_receiver():
	var btn = SuperButton.new()
	watch_signals(btn)
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child_autofree(btn)

	var sender = InputSender.new(Input)
	sender.mouse_warp = true

	var start_pos = Vector2i(25, 75)
	for i in 15:
		var new_pos = start_pos + Vector2i(i * 10, 0)
		await sender.wait(.1)\
			.mouse_left_button_down(new_pos)\
			.hold_for(.1)\
			.wait(.1).idle

	_print_emitted_signals(btn)
	assert_signal_emitted(btn, 'pressed')
	assert_signal_emitted(btn, 'button_down')
	assert_signal_emitted(btn, 'button_up')
	assert_signal_emitted(btn, 'gui_input')


func test_clicking_two_buttons_triggers_focus_events():
	var btn = SuperButton.new()
	watch_signals(btn)
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child_autofree(btn)

	var btn2 = SuperButton.new()
	watch_signals(btn2)
	btn2.size = Vector2(100, 100)
	btn2.position = Vector2(160, 50)
	add_child_autofree(btn2)

	var sender = InputSender.new(Input)
	sender.mouse_warp = true

	var start_pos = Vector2(100, 75)
	for i in 10:
		var new_pos = start_pos + Vector2(i * 10, 0)
		await sender.mouse_left_click_at(new_pos).idle

	_print_emitted_signals(btn)
	_print_emitted_signals(btn2)



func test_clicking_things_with_button_as_receiver():
	var btn = SuperButton.new()
	watch_signals(btn)
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child_autofree(btn)

	var sender = InputSender.new(btn)
	sender.mouse_warp = true

	var start_pos = Vector2i(25, 75)
	for i in 15:
		var new_pos = start_pos + Vector2i(i * 10, 0)
		await sender.wait(.1)\
			.mouse_left_button_down(new_pos)\
			.hold_for(.1)\
			.wait(.1).idle

	_print_emitted_signals(btn)
	assert_signal_not_emitted(btn, 'pressed')
	assert_signal_not_emitted(btn, 'button_down')
	assert_signal_not_emitted(btn, 'button_up')
	assert_signal_not_emitted(btn, 'gui_input')

