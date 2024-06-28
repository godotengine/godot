extends HBoxContainer


signal pressed
signal value_changed


const ICON_PICKER: String = "res://addons/terrain_3d/icons/picker.svg"
const ICON_PICKER_CHECKED: String = "res://addons/terrain_3d/icons/picker_checked.svg"
const MAX_POINTS: int = 2


var icon_picker: Texture2D
var icon_picker_checked: Texture2D
var points: PackedVector3Array
var picking_index: int = -1


func _init() -> void:
	icon_picker = load(ICON_PICKER)
	icon_picker_checked = load(ICON_PICKER_CHECKED)
	
	points.resize(MAX_POINTS)
	
	for i in range(MAX_POINTS):
		var button := Button.new()
		button.icon = icon_picker
		button.tooltip_text = "Pick point on the Terrain"
		button.set_meta(&"point_index", i)
		button.pressed.connect(_on_button_pressed.bind(i))
		add_child(button)
	
	_update_buttons()


func _on_button_pressed(button_index: int) -> void:
	points[button_index] = Vector3.ZERO
	picking_index = button_index
	_update_buttons()
	pressed.emit()


func _update_buttons() -> void:
	for child in get_children():
		if child is Button:
			_update_button(child)


func _update_button(button: Button) -> void:
	var index: int = button.get_meta(&"point_index")
	
	if points[index] != Vector3.ZERO:
		button.icon = icon_picker_checked
	else:
		button.icon = icon_picker


func clear() -> void:
	points.fill(Vector3.ZERO)
	_update_buttons()
	value_changed.emit()


func all_points_selected() -> bool:
	return points.count(Vector3.ZERO) == 0


func add_point(p_value: Vector3) -> void:
	if points.has(p_value):
		return
	
	# If manually selecting a point individually
	if picking_index != -1:
		points[picking_index] = p_value
		picking_index = -1
	else:
		# Else picking a sequence of points (non-drawable)
		for i in range(MAX_POINTS):
			if points[i] == Vector3.ZERO:
				points[i] = p_value
				break
	_update_buttons()
	value_changed.emit()


func get_points() -> PackedVector3Array:
	return points
