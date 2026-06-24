# main.gd — UWP embedding smoke test.
# Spinning cube proves rendering; clicking anywhere recolors the cube and
# updates the label, proving host->engine input injection works.

extends Node3D

var _clicks := 0

@onready var _cube: CSGBox3D = $Cube
@onready var _label: Label = $UI/InfoLabel


func _ready() -> void:
	print("[UWP Embed Test] _ready — rendering driver: ",
			RenderingServer.get_current_rendering_driver_name())
	print("[UWP Embed Test] display server: ", DisplayServer.get_name())
	print("[UWP Embed Test] window size: ", DisplayServer.window_get_size())


func _process(delta: float) -> void:
	_cube.rotate_y(delta * 1.5)
	_cube.rotate_x(delta * 0.4)
	_label.text = "Godot %s in UWP SwapChainPanel\nFPS: %d  |  Size: %s  |  Clicks: %d" % [
		Engine.get_version_info()["string"],
		Engine.get_frames_per_second(),
		DisplayServer.window_get_size(),
		_clicks,
	]


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		_clicks += 1
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(randf(), randf(), randf())
		_cube.material = mat
		print("[UWP Embed Test] click #", _clicks, " at ", event.position)
