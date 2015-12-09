
extends ColorPickerButton


func get_drag_data(pos):
	# Use another colorpicker as drag preview
	var cpb = ColorPickerButton.new()
	cpb.set_color(get_color())
	cpb.set_size(Vector2(50, 50))
	set_drag_preview(cpb)
	# Return color as drag data
	return get_color()


func can_drop_data(pos, data):
	return typeof(data) == TYPE_COLOR


func drop_data(pos, data):
	set_color(data)
