@tool
extends PanelContainer

signal brush_selected
signal brush_edited
signal brush_removed

@onready var label = find_child("label")
@onready var edit= find_child("edit")
@onready var remove = find_child("remove")
var color
var brush_type = 0
var brush_data = {}
var brush_layer 

func _ready():	
	label.pressed.connect(func(): brush_selected.emit())
	visibility_changed.connect(on_resize)
	resized.connect(on_resize)
	
func set_height_brush(bname, bicon):
	edit.free()
	remove.free()
	label.text = bname
	label.icon = bicon
	
func set_color_brush(layer_group, i):
	brush_layer = layer_group
	var brush = layer_group.layers[i]
	label.text = brush.NAME
	label.icon = load(brush.ICON) if FileAccess.file_exists(brush.ICON) else null
	if layer_group.brush_name == "Color Paint":	
		brush_data = {"color": brush.color, "hardness":brush.hardness }
		brush_type = 0
	elif layer_group.brush_name == "Channel Painter": 
		brush_data = {
			"hardness":brush.hardness,
			"r_on": brush.red, 
			"r_value":brush['red-value'],
			"g_on": brush.green, 
			"g_value":brush['green-value'],
			"b_on": brush.blue, 
			"b_value":brush['blue-value'],
			"a_on": brush.alpha, 
			"a_value":brush['alpha-value'],
			 }
		brush_type = 1
	elif layer_group.brush_name == "Bitwise Brush": 
		brush_data = {"bit": brush.bit, "value":brush.value }
		brush_type = 2
	elif layer_group.brush_name == "Paint 16": 
		brush_data = {"paint16layer": brush['paint-layer'] }
		brush_type = 3
	elif layer_group.brush_name == "Paint 256": 
		brush_data = {"paint256layer": brush['paint-layer'] }
		brush_type = 4		
				
	if brush_type == 0:
		color = brush.color
		var stylebox = preload("res://addons/m_terrain/gui/styles/button_stylebox.tres").duplicate()
		stylebox.bg_color = color		
		label.set("theme_override_styles/normal", stylebox)
		label.set("theme_override_styles/focus", stylebox)
		label.set("theme_override_styles/hover", stylebox)
		label.set("theme_override_styles/pressed", stylebox)
	else:		
		label.set("theme_override_styles/normal", null)
		label.set("theme_override_styles/focus", null)
		label.set("theme_override_styles/hover", null)
		label.set("theme_override_styles/pressed", null)	
	edit.pressed.connect(edit_brush)
	edit.tooltip_text = str("edit ", brush.NAME)
	if layer_group.layers[i].NAME == "background":
		remove.disabled = true
		remove.tooltip_text = "cannot remove background brush"
	else:
		remove.pressed.connect(remove_brush)	

func set_text_brush(text):
	label.text = text
	label.icon = null
	edit.queue_free()
	remove.queue_free()

func edit_brush():
	var popup = preload("res://addons/m_terrain/gui/mtools_create_color_brush.tscn").instantiate()
	add_child(popup)	
	var bicon = label.icon.resource_path if label.icon else ""
	popup.load_brush(brush_layer, label.text, bicon, brush_data)
	popup.brush_created.connect(
		func(new_name, new_icon_path, data): 				
			brush_edited.emit(new_name, new_icon_path, data, get_index())
	)

func remove_brush():
	var popup = preload("res://addons/m_terrain/gui/mtools_layer_warning_popup.tscn").instantiate()
	add_child(popup)
	popup.confirmed.connect(func():
		brush_removed.emit(get_index())
		queue_free()		
		get_parent().remove_child(self)
		
		
	)

func on_resize():
	label.set("theme_override_constants/icon_max_width", theme.default_font_size*2)
