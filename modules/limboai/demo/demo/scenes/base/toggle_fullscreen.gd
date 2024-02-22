extends Button


func _ready() -> void:
	pressed.connect(_toggle_fullscreen)

	# Set scaling properties and base resolution
	get_window().mode = Window.MODE_FULLSCREEN
	get_window().content_scale_aspect = Window.CONTENT_SCALE_ASPECT_KEEP
	get_window().content_scale_mode = Window.CONTENT_SCALE_MODE_CANVAS_ITEMS
	get_window().content_scale_size = Vector2i(1920,1080)


func _toggle_fullscreen() -> void:
	if get_window().mode != Window.MODE_FULLSCREEN:
		get_window().mode = Window.MODE_FULLSCREEN
	else:
		get_window().mode = Window.MODE_MAXIMIZED
