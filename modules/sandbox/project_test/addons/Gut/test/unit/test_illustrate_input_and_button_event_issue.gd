extends GutTest


class PrintEventsButton:
	extends Button

	func _ready():
		button_down.connect(_on_button_down)
		button_up.connect(_on_button_up)
		print('button ready')

	func _on_button_down():
		print('    button down emitted')

	func _on_button_up():
		print('    button up emitted')

	func _gui_input(event):
		print('    _gui_input:  ', event)

	func _input(event):
		print('    _input:  ', event)

	func _unhandled_input(event):
		print('    _unhandled_input:  ', event)

	func print_has_method(method_name):
		print("self has = ", has_method(method_name))
		print('super has = ', super.has_method(method_name))


# var _sender = InputSender.new(Input)

# func _before_each():
# 	_sender.wait_frames(1)
# 	await _sender.idle

# func _after_each():
# 	_sender.release_all()
# 	_sender.clear()

func before_all():
	register_inner_classes(load('res://test/unit/test_illustrate_input_and_button_event_issue.gd'))

func test_something():
	var btn = autofree(PrintEventsButton.new())
	btn.print_has_method('_input')
	btn._input(InputEventMouseButton.new())


func illustrate(btn):
	var button_down = InputEventMouseButton.new()
	button_down.button_index = MOUSE_BUTTON_LEFT
	button_down.pressed = true
	button_down.position = btn.position + Vector2(10, 10)

	var button_up = InputEventMouseButton.new()
	button_up.button_index = MOUSE_BUTTON_LEFT
	button_up.pressed = false
	button_up.position = btn.position + Vector2(10, 10)

	print('-- sending button down')
	Input.parse_input_event(button_down)

	await get_tree().create_timer(.2).timeout

	print('-- sending button up')
	Input.parse_input_event(button_up)

	await get_tree().create_timer(.2).timeout
	print('-- done')

	return btn

func test_illustrate():
	# gut.get_doubler().print_source = true
	var btn = partial_double(PrintEventsButton, DOUBLE_STRATEGY.SCRIPT_ONLY).new()
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child(btn)
	watch_signals(btn)

	# If we use get_tree().root when calling illustrate, then all the signal
	# asserts in other tests will fail and I have no idea why.
	await wait_physics_frames(10)
	await illustrate(btn)
	assert_called(btn, '_on_button_up')
	assert_signal_emitted(btn, 'button_down')
	assert_signal_emitted(btn, 'button_up')


func test_same_thing_the_gut_way():
	var btn = PrintEventsButton.new()
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child_autofree(btn)
	watch_signals(btn)

	var sender = InputSender.new(Input)

	await sender\
		.mouse_left_click_at(btn.position + Vector2(10, 10), '10f')\
		.idle
	assert_signal_emitted(btn, 'button_down')
	assert_signal_emitted(btn, 'button_up')
	# GutUtils.pretty_print(_signal_watcher._watched_signals)


func test_same_thing_another_way():
	var btn = PrintEventsButton.new()
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child_autofree(btn)
	watch_signals(btn)

	var sender = InputSender.new(Input)
	await sender\
		.mouse_left_button_down(btn.position + Vector2(10, 10))\
		.hold_for('5f')\
		.wait_frames(5)\
		.idle

	assert_signal_emitted(btn, 'button_down')
	assert_signal_emitted(btn, 'button_up')


func test_same_thing_another_way_but_with_autoflush():
	# await wait_frames(30)
	var btn = PrintEventsButton.new()
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_child_autofree(btn)
	watch_signals(btn)

	var sender = InputSender.new(Input)
	sender.set_auto_flush_input(true)

	await sender\
		.mouse_left_button_down(btn.position + Vector2(10, 10))\
		.hold_for('1f')\
		.idle

	assert_signal_emitted(btn, 'button_down')
	assert_signal_emitted(btn, 'button_up')
