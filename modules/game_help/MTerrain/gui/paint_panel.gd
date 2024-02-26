@tool
extends VBoxContainer
class_name MPaintPanel

@onready var brush_type_checkbox:=$brush_type
@onready var grass_add_checkbox:=$grass_add
@onready var brush_list_option:=$brush_list
@onready var brush_slider:=$brush_size/brush_slider
@onready var brush_lable:=$brush_size/lable
@onready var heightmap_layers:=$heightmap_layers
@onready var layer_buttons:=$layer_buttons
@onready var add_name_line:=$layer_buttons/addName
@onready var color_brush_root:=$color_brushe_scroll/color_brushe_root
@onready var color_brush_scroll:=$color_brushe_scroll
@onready var brush_masks:=$brush_masks
@onready var mask_cutt_off_label:=$HBoxContainer2/Labelmask_cutt_off_label
@onready var mask_cutt_off_slider:=$HBoxContainer2/mask_cutt_off_slider

var brush_layers_res = preload("res://addons/m_terrain/gui/brush_layers.tscn")
var layers_title_res = preload("res://addons/m_terrain/gui/layers_title.tscn")

var hide_icon = preload("res://addons/m_terrain/icons/hidden.png")
var show_icon = preload("res://addons/m_terrain/icons/show.png")

signal brush_size_changed

var float_prop_element=preload("res://addons/m_terrain/gui/control_prop_element/float.tscn")
var float_range_prop_element=preload("res://addons/m_terrain/gui/control_prop_element/float_range.tscn")
var bool_element = preload("res://addons/m_terrain/gui/control_prop_element/bool.tscn")
var int_element = preload("res://addons/m_terrain/gui/control_prop_element/int.tscn")
var int_enum_element = preload("res://addons/m_terrain/gui/control_prop_element/int_enum.tscn")

var brush_manager:MBrushManager = MBrushManager.new()
var is_color_brush:=true
var brush_id:int=-1
var active_heightmap_layer:="background"
var active_terrain:MTerrain

var brush_size:float

var property_element_list:Array
var color_brush_layers:Array
var color_brush_titles:Array
var selected_layer_group=0
var current_uniform:=""
var current_color_brush:=""

var is_grass_add:bool = true


var smooth_brush_id:int
var raise_brush_id:int
var to_height_brush_id:int
var hole_brush_id:int

func _ready():
	_on_brush_type_toggled(false)
	change_brush_size(50)
	smooth_brush_id = brush_manager.get_height_brush_id("Smooth")
	raise_brush_id = brush_manager.get_height_brush_id("Raise")
	to_height_brush_id = brush_manager.get_height_brush_id("To Height")
	hole_brush_id = brush_manager.get_height_brush_id("Hole")

func set_active_terrain(input:MTerrain):
	active_terrain = input
	update_heightmap_layers()
	set_mask_cutoff_value(mask_cutt_off_slider.value)

func set_grass_mode(input:bool):
	for e in property_element_list:
		e.visible = not input
	brush_list_option.visible = not input
	heightmap_layers.visible = not input
	layer_buttons.visible = not input
	brush_type_checkbox.visible = not input
	$layer_note.visible = not input
	grass_add_checkbox.visible = input
	if is_color_brush:
		color_brush_scroll.visible = false
	$grass_lable.visible = input
	if not input:
		_on_brush_type_toggled(is_color_brush)
	


func _on_brush_type_toggled(button_pressed):
	is_color_brush = button_pressed
	brush_list_option.clear()
	brush_list_option.visible = not button_pressed
	layer_buttons.visible = not button_pressed
	heightmap_layers.visible = not button_pressed
	$layer_note.visible = not button_pressed
	color_brush_root.visible = button_pressed
	color_brush_scroll.visible = button_pressed
	if button_pressed:
		brush_type_checkbox.text = "Color brush"
		_on_brush_list_item_selected(-1)
		create_color_brush_layers()
	else:
		remove_color_brush_layers()
		brush_type_checkbox.text = "Height brush"
		var brushe_names = brush_manager.get_height_brush_list()
		for n in brushe_names:
			brush_list_option.add_item(n)
			_on_brush_list_item_selected(0)

func create_color_brush_layers():
	remove_color_brush_layers()
	var layers_group:Array = active_terrain.get_layers_info()
	var layer_group = 0
	for layers in layers_group:
		var title_lable = layers_title_res.instantiate()
		color_brush_titles.push_back(title_lable)
		var title :String= layers["title"]
		if title.is_empty():
			title = "Layer Group "+str(layer_group)
		title_lable.text = title
		color_brush_root.add_child(title_lable)
		var l = brush_layers_res.instantiate()
		color_brush_layers.push_back(l)
		color_brush_root.add_child(l)
		l.create_layers(layers["info"])
		l.index = layers["index"]
		l.uniform = layers["uniform"]
		l.brush_name = layers["brush_name"]
		l.connect("item_selected",Callable(self,"brush_layer_selected").bind(layer_group))
		if layer_group==0 and layers.size()!=0:
			l.select(0)
			brush_layer_selected(0, layer_group)
		layer_group += 1

func remove_color_brush_layers():
	for l in color_brush_layers:
		l.queue_free()
	for l in color_brush_titles:
		l.queue_free()
	color_brush_layers.clear()
	color_brush_titles.clear()

func brush_layer_selected(index, layer_group):
	if selected_layer_group != layer_group:
		var i=0
		for l in color_brush_layers:
			if l.is_class("ItemList"):
				if i != layer_group:
					l.deselect_all()
				i +=1
	selected_layer_group = layer_group
	var l = color_brush_layers[layer_group]
	current_uniform = l.uniform
	current_color_brush = l.brush_name
	var group_index = l.index
	active_terrain.set_color_layer(index,group_index,l.brush_name)


func _on_brush_list_item_selected(index):
	clear_property_element()
	if index < -1: return
	brush_id = index
	var brush_props:Array
	if is_color_brush:
		pass
	else:
		brush_props = brush_manager.get_height_brush_property(brush_id)
	for p in brush_props:
		create_props(p)



func create_props(dic:Dictionary):
	var element
	if dic["type"]==TYPE_FLOAT:
		var rng = dic["max"] - dic["min"]
		if dic["hint"] == "range":
			element = float_range_prop_element.instantiate()
			element.set_min(dic["min"])
			element.set_max(dic["max"])
			element.set_step(dic["hint_string"].to_float())
		else:
			element = float_prop_element.instantiate()
			element.min = dic["min"]
			element.max = dic["max"]
	elif dic["type"]==TYPE_BOOL:
		element = bool_element.instantiate()
	elif dic["type"]==TYPE_INT:
		if dic["hint"] == "enum":
			element = int_enum_element.instantiate()
			element.set_options(dic["hint_string"])
		else:
			element = int_element.instantiate()
			element.set_min(dic["min"])
			element.set_max(dic["max"])
	add_child(element)
	element.connect("prop_changed",Callable(self,"prop_change"))
	element.set_value(dic["default_value"])
	element.set_name(dic["name"])
	property_element_list.append(element)



func clear_property_element():
	for e in property_element_list:
		if is_instance_valid(e):
			e.queue_free()
	property_element_list = []

func prop_change(prop_name,value):
	if is_color_brush:
		pass
	else:
		brush_manager.set_height_brush_property(prop_name,value,brush_id)


func change_brush_size(value):
	brush_slider.value = value

func _on_brush_slider_value_changed(value):
	brush_size = value
	brush_slider.max_value = 100*pow(value,0.3)
	brush_lable.text = "brush size "+str(value).pad_decimals(1)
	emit_signal("brush_size_changed",value)



### Layers
func update_heightmap_layers(select=0):
	if not active_terrain: return
	heightmap_layers.clear()
	var inputs = active_terrain.get_heightmap_layers()
	for i in range(0,inputs.size()):
		heightmap_layers.add_item(inputs[i])
		if i!=0: # We don't add visibilty icon for background layer
			var visibile = active_terrain.get_layer_visibility(inputs[i])
			if visibile:
				heightmap_layers.set_item_icon(i,show_icon)
			else:
				heightmap_layers.set_item_icon(i,hide_icon)
	heightmap_layers.select(select)
	

func set_active_layer():
	var selected:PackedInt32Array= heightmap_layers.get_selected_items()
	if selected.size() == 0:
		active_terrain.set_active_layer_by_name("background")
		return
	var lname = heightmap_layers.get_item_text(selected[0])
	active_terrain.set_active_layer_by_name(lname)


func _on_heightmap_layer_item_selected(index):
	active_heightmap_layer = heightmap_layers.get_item_text(index)
	if not active_terrain:
		printerr("No active terrain")
		return
	active_terrain.set_active_layer_by_name(active_heightmap_layer)
	if active_heightmap_layer == "holes":
		brush_list_option.select(hole_brush_id)
		_on_brush_list_item_selected(hole_brush_id)
	elif active_heightmap_layer != "holes" and brush_id == hole_brush_id:
		brush_list_option.select(raise_brush_id)
		_on_brush_list_item_selected(raise_brush_id)

func _on_merge_bt_pressed():
	set_active_layer()
	active_terrain.merge_heightmap_layer()
	update_heightmap_layers()

func _on_add_bt_pressed():
	if not active_terrain:
		printerr("No active terrain")
		return
	var layer_name:String= add_name_line.text
	if layer_name.is_empty():
		printerr("Layer Name is empty")
		return
	add_name_line.text = ""
	if layer_name.is_empty():
		printerr("Layer name is empty")
	active_terrain.add_heightmap_layer(layer_name)
	update_heightmap_layers()


func _on_add_name_gui_input(event):
	if event is InputEventKey:
		if event.keycode == KEY_ENTER and not event.echo and event.pressed:
			_on_add_bt_pressed()

func _on_remove_bt_pressed():
	if not active_terrain:
		printerr("No active terrain")
	set_active_layer()
	active_terrain.remove_heightmap_layer()
	update_heightmap_layers()
	

func _on_visibilty_bt_pressed():
	if not active_terrain:
		printerr("No active terrain")
	var selected:PackedInt32Array= heightmap_layers.get_selected_items()
	if selected.size()==0:
		return
	set_active_layer()
	active_terrain.toggle_heightmap_layer_visibile()
	update_heightmap_layers(selected[0])


func _on_grass_add_toggled(button_pressed):
	is_grass_add = button_pressed


var last_height_brush_id:int

func _input(event):
	if event is InputEventKey:
		if not is_color_brush:
			last_height_brush_id = brush_list_option.get_selected_id()
			if event.keycode == KEY_SHIFT:
				if event.is_pressed():
					if not event.echo:
						brush_id = smooth_brush_id
				else:
					brush_id = last_height_brush_id
			elif event.keycode == KEY_CTRL:
				if event.is_pressed():
					if not event.echo:
						brush_id = to_height_brush_id
				else:
					brush_id = last_height_brush_id
		if event.keycode == KEY_SHIFT:
			if event.is_pressed():
				if not event.echo:
					is_grass_add = not is_grass_add
			else:
				is_grass_add = not is_grass_add

func set_mask_cutoff_value(value):
	if active_terrain:
		active_terrain.set_mask_cutoff(value)
	mask_cutt_off_label.text = "Mask Cutoff: " + str(value)
	
