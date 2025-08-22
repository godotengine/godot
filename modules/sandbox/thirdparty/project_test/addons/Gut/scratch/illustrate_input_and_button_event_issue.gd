extends SceneTree

class PrintEventsButton:
	extends Button

	func _ready():
		button_down.connect(_on_button_down)
		button_up.connect(_on_button_up)

	func _on_button_down():
		print('    button down')

	func _on_button_up():
		print('    button up')

	func _gui_input(event):
		print('    _gui_input:  ', event)

	func _input(event):
		print('    _input:  ', event)

	func _unhandled_input(event):
		print('    _unhandled_input:  ', event)



func illustrate(add_to):
	var btn = PrintEventsButton.new()
	btn.size = Vector2(100, 100)
	btn.position = Vector2(50, 50)
	add_to.add_child(btn)

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

	await get_root().get_tree().create_timer(.2).timeout

	print('-- sending button up')
	Input.parse_input_event(button_up)

	await get_root().get_tree().create_timer(.2).timeout
	print('-- done')


func _make_the_whole_gut_tree_to_add_to():
	var gut_runner = load('res://addons/gut/gui/GutRunner.tscn').instantiate()
	get_root().add_child(gut_runner)

	var gut = gut_runner.get_gut()
	gut_runner._gut_layer.add_child(gut)
	gut.add_children_to = gut_runner

	var add_to = GutTest.new()
	gut.add_child(add_to)
	await get_root().get_tree().create_timer(.2).timeout
	return add_to


func _init():
	await get_root().get_tree().create_timer(.2).timeout
	# var add_to = await _make_the_whole_gut_tree_to_add_to()
	var add_to = get_root()

	await illustrate(add_to)
	print("well, shit...it looks like it is fine")
	quit()