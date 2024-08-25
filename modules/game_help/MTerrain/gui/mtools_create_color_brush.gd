@tool
extends Window
signal brush_created
@onready var create_button = find_child("create")
@onready var brush_name = find_child("brush_name")
@onready var icon = find_child("icon_path")
@onready var hardness = find_child("hardness")
@onready var color = find_child("color_picker")
@onready var r = find_child("ChannelR")
@onready var g = find_child("ChannelG")
@onready var b = find_child("ChannelB")
@onready var a = find_child("ChannelA")
@onready var bitwise = find_child("Bitwise")
@onready var paint16 = find_child("Paint16")
@onready var paint256 = find_child("Paint256")
@onready var brush_type = find_child("brush_type")
var existing_brush_names = []
var is_update_mode = false

func _ready():
	brush_name.text_changed.connect(validate_brush_name)
	create_button.pressed.connect(func():		
		brush_created.emit(brush_name.text, icon.text, get_data()) 
		queue_free()
	)
	close_requested.connect(queue_free)	
	color.pressed.connect(func(): 
		var picker = color.get_picker().get_parent()
		picker.position.x+= picker.size.x
		picker.position.y = color.global_position.y + position.y
		)
	var load_icon_button = find_child("load_icon_button")
	load_icon_button.pressed.connect(func():
		var popup = FileDialog.new()
		popup.file_mode = FileDialog.FILE_MODE_OPEN_FILE
		popup.use_native_dialog = true
		popup.filters = PackedStringArray(["*.png, *.jpg ; Image Files"])
		add_child(popup)
		popup.show()
		popup.size = size
		popup.position = position
		popup.file_selected.connect(func(file): icon.text = file)
	)

func validate_brush_name(new_name):
	if new_name in existing_brush_names:
		create_button.disabled = true
		create_button.text = "please pick a different brush name"
	else:
		create_button.disabled = false
		create_button.text = "Update" if is_update_mode else "Create"
		

func get_data():
	if color.visible:
		return {"color": color.color, "hardness": hardness.value}
	elif r.visible:
		return { 
			"r_on": r.button.button_pressed, 
			"r_value": r.slider.value,
			"g_on": g.button.button_pressed, 
			"g_value": g.slider.value,
			"b_on": b.button.button_pressed, 
			"b_value": b.slider.value,
			"a_on": a.button.button_pressed, 
			"a_value": a.slider.value,			
			"hardness": hardness.value,
		}
	elif bitwise.visible:
		return { "bit": bitwise.slider.value, "bit_value": bitwise.button.button_pressed}
	elif paint16.visible:
		return {"paint16layer": paint16.slider.value}
	elif paint256.visible:
		return {"paint256layer": paint256.slider.value}
		
func clear_options():
	hardness.visible = false
	color.visible = false
	r.visible = false
	g.visible = false
	b.visible = false
	a.visible = false
	bitwise.visible = false
	paint16.visible = false
	paint256.visible = false
	hardness.visible = false
	
func init_for_color(existing_brushes):	
	clear_options()
	color.visible = true
	hardness.visible = true
	existing_brush_names = existing_brushes
	brush_type.text = "Color Brush Settings"

func init_for_channel(existing_brushes):
	clear_options()
	r.visible = true
	g.visible = true
	b.visible = true
	a.visible = true
	hardness.visible = true
	existing_brush_names = existing_brushes
	brush_type.text = "Channel Brush Settings"
	
func init_for_bitwise(existing_brushes):
	clear_options()	
	bitwise.visible = true
	existing_brush_names = existing_brushes
	brush_type.text = "Bitwise Brush Settings"
	
func init_for_16(existing_brushes):
	clear_options()	
	paint16.visible = true
	existing_brush_names = existing_brushes
	brush_type.text = "Paint16 Brush Settings"
	
func init_for_256(existing_brushes):
	clear_options()	
	paint256.visible = true
	existing_brush_names = existing_brushes
	brush_type.text = "Paint256 Brush Settings"

func load_brush(layer_group, bname, bicon, data={}):
	var existing_brush_names = layer_group.layers.map(func(a): return a.NAME).filter(func(a): return a != bname)
	brush_name.text = bname
	if bname == "background":
		brush_name.editable = false
	icon.text = bicon
	if layer_group.brush_name == "Color Paint":
		init_for_color(existing_brush_names)	
		color.color = data.color
		hardness.value = data.hardness
	if layer_group.brush_name == "Channel Painter":	
		init_for_channel(existing_brush_names)			
		hardness.value = data.hardness
		r.button.button_pressed = data.r_on
		r.value_changed(data.r_value)
		g.button.button_pressed = data.g_on
		g.value_changed(data.g_value)		
		b.button.button_pressed = data.b_on
		b.value_changed(data.b_value)		
		a.button.button_pressed = data.a_on
		a.value_changed(data.a_value)
	if layer_group.brush_name == "Bitwise Brush":
		init_for_bitwise(existing_brush_names)			
		bitwise.button.button_pressed = data.bit_value
		bitwise.value_changed(data.bit)
	if layer_group.brush_name == "Paint 16":
		init_for_16(existing_brush_names)			
		paint16.value_changed(data.paint16Layer)
	if layer_group.brush_name == "Paint 256":
		init_for_256(existing_brush_names)			
		paint256.value_changed(data.paint256Layer)
	create_button.text = "Update"
	is_update_mode = true

