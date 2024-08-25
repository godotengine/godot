#BRUSH button
@tool
extends Button

@onready var brush_container = find_child("brush_container")
@onready var brush_settings_container = find_child("brush_settings")

var height_brush_id = 0

var last_height_brush_id
var is_grass_add = true
var brush_mode = &"" # &"sculpt", &"paint", &"grass", &"navigation"
var active_terrain #used for setting active color brush

var float_prop_element=preload("res://addons/m_terrain/gui/control_prop_element/float.tscn")
var float_range_prop_element=preload("res://addons/m_terrain/gui/control_prop_element/float_range.tscn")
var bool_element = preload("res://addons/m_terrain/gui/control_prop_element/bool.tscn")
var int_element = preload("res://addons/m_terrain/gui/control_prop_element/int.tscn")
var int_enum_element = preload("res://addons/m_terrain/gui/control_prop_element/int_enum.tscn")


var height_brush_manager

var smooth_brush_id
var raise_brush_id
var to_height_brush_id
var hole_brush_id
var reverse_property_control

var property_element_list:Array

var color_brush_layer: MBrushLayers
var color_layer_group_id:int
var color_brush_uniform:String
var color_brush_name:String # "Color Paint", "Channel Painter", "Bitwise Brush", "Paint 16", Paint 256" 

var no_image = preload("res://addons/m_terrain/icons/no_images.png") #For color brush


@onready var add_color_brush_button = find_child("add_color_brush_button")

func _ready():
	var panel = get_child(0)
	panel.visible = false
	panel.position.y = -panel.size.y
	panel.gui_input.connect(fix_gui_input)
	panel.size.x = get_viewport().size.x - global_position.x
	panel.visibility_changed.connect(_on_panel_visibility_changed)
	add_color_brush_button.pressed.connect(show_add_color_brush_popup)
	
func fix_gui_input(event: InputEvent):
	if event is InputEventMouseButton:
		get_viewport().set_input_as_handled()
	
func _toggled(toggled_on):
	if toggled_on:
		get_child(0).size.x = get_viewport().size.x - global_position.x
 
func clear_brushes():
	#brush_container.clear()	
	for child in brush_container.get_children():
		brush_container.remove_child(child)
		child.queue_free()
	clear_property_element()
	set("theme_override_styles/normal", null)
	text = ""
	for connection in brush_container.get_signal_connection_list("brush_selected"):
		connection.signal.disconnect(connection.callable)	
	#for connection in brush_container.get_signal_connection_list("item_clicked"):
	#	connection.signal.disconnect(connection.callable)
	add_color_brush_button.visible = false
	find_child("brush_settings_panel").visible = false
	
	
#region Height Brushes
func init_height_brushes(new_brush_manager):		
	clear_brushes()	
	find_child("brush_settings_panel").visible = true
	
	brush_mode = &"sculpt"	
	#brush_container.brush_selected.connect(on_height_brush_select)	
	height_brush_manager = new_brush_manager
	smooth_brush_id = height_brush_manager.get_height_brush_id("Smooth")
	raise_brush_id = height_brush_manager.get_height_brush_id("Raise")
	to_height_brush_id = height_brush_manager.get_height_brush_id("To Height")
	hole_brush_id = height_brush_manager.get_height_brush_id("Hole")

	var brush_names = height_brush_manager.get_height_brush_list()
	
	for n in brush_names:		
		var brush_item = preload("res://addons/m_terrain/gui/mtools_brush_item.tscn").instantiate()
		brush_container.add_child(brush_item)				
		if "raise" in n.to_lower():			
			brush_item.set_height_brush(n, preload("res://addons/m_terrain/icons/brush_icon_raise.svg"))
		if "height" in n.to_lower():
			brush_item.set_height_brush(n, preload("res://addons/m_terrain/icons/brush_icon_to_height.svg"))
		if "smooth" in n.to_lower():
			brush_item.set_height_brush(n, preload("res://addons/m_terrain/icons/brush_icon_smooth.svg"))
		if "hole" in n.to_lower():
			brush_item.set_height_brush(n, preload("res://addons/m_terrain/icons/brush_icon_hole.svg"))	
		if "layer" in n.to_lower():
			brush_item.set_height_brush(n, preload("res://addons/m_terrain/icons/eraser_icon.svg"))	
			
		brush_item.brush_selected.connect(on_height_brush_select.bind(brush_item))
	on_height_brush_select(brush_container.get_child(0))	
	
func on_height_brush_select(item):
	clear_property_element()	
	height_brush_id = item.get_index()
	var brush_props = height_brush_manager.get_height_brush_property(height_brush_id)
	for p in brush_props:
		create_props(p)
	text = ""
	set("theme_override_styles/normal", null)
	icon = item.label.icon
	tooltip_text = "Current brush: " + item.label.text

func create_props(dic:Dictionary):
	var element
	if dic["type"]==TYPE_FLOAT:
		var rng = dic["max"] - dic["min"]
		if dic["hint"] == "range":
			element = float_range_prop_element.instantiate()
			brush_settings_container.add_child(element)
			element.set_min(dic["min"])
			element.set_max(dic["max"])
			element.set_step(dic["hint_string"].to_float())
		else:
			element = float_prop_element.instantiate()
			brush_settings_container.add_child(element)
			element.min = dic["min"]
			element.max = dic["max"]
	elif dic["type"]==TYPE_BOOL:
		element = bool_element.instantiate()
		brush_settings_container.add_child(element)
	elif dic["type"]==TYPE_INT:
		if dic["hint"] == "enum":
			element = int_enum_element.instantiate()
			brush_settings_container.add_child(element)
			element.set_options(dic["hint_string"])
		else:
			element = int_element.instantiate()
			brush_settings_container.add_child(element)
			element.set_min(dic["min"])
			element.set_max(dic["max"])
	element.prop_changed.connect(prop_change)
	element.set_value(dic["default_value"])
	element.set_name(dic["name"])
	property_element_list.append(element)	
	if element.prop_name.to_lower() == "revers":
		reverse_property_control = element

func clear_property_element():
	for e in property_element_list:
		if is_instance_valid(e):
			e.queue_free()
	property_element_list = []

func prop_change(prop_name,value):	
	height_brush_manager.set_height_brush_property(prop_name,value,height_brush_id)

#endregion

#region Color Brushes
func init_color_brushes(terrain: MTerrain = null, layer_group_id=0):	
	clear_brushes()
	active_terrain = terrain
	brush_mode = &"paint"
	if terrain == null:
		return	
							
	add_color_brush_button.visible = true
	color_brush_layer = active_terrain.brush_layers[layer_group_id]						
		
	for i in color_brush_layer.layers.size():	
		var brush = color_brush_layer.layers[i]					
		var bname = brush.NAME
		if bname.is_empty():
			bname = str("layer ", i)
		var bicon:Texture = load(brush.ICON) if FileAccess.file_exists(brush.ICON) else null						
		
		var brush_item = preload("res://addons/m_terrain/gui/mtools_brush_item.tscn").instantiate()		
		brush_container.add_child(brush_item)		
		brush_item.set_color_brush(color_brush_layer, i)
		brush_item.brush_selected.connect( brush_layer_selected.bind(brush_item.get_index(), color_brush_layer))
		brush_item.brush_edited.connect(update_color_brush)
		brush_item.brush_removed.connect(remove_color_brush)
	
	color_layer_group_id = layer_group_id
	color_brush_uniform = color_brush_layer.uniform_name
	color_brush_name = color_brush_layer.brush_name
	
	if brush_container.get_child_count() != 0:
		brush_layer_selected(0, layer_group_id)

func brush_layer_selected(index, layer_group):			
	active_terrain.set_color_layer(index, color_layer_group_id,color_brush_name)
	var brush_icon = brush_container.get_child(index).label.icon
	if brush_icon and brush_icon != no_image:
		icon = brush_icon
	else:
		icon = null
	text = "" #brush_container.get_item_text(index)
	tooltip_text = "Current brush: " + brush_container.get_child(index).label.text
	var color = brush_container.get_child(index).color
	if color:
		var stylebox = StyleBoxFlat.new()
		stylebox.bg_color = color
		set("theme_override_styles/normal", stylebox)
		set("theme_override_styles/focus", stylebox)
		set("theme_override_styles/hover", stylebox)
		set("theme_override_styles/pressed", stylebox)
	else:
		set("theme_override_styles/normal", null)
		set("theme_override_styles/focus", null)
		set("theme_override_styles/hover", null)
		set("theme_override_styles/pressed", null)
		

func show_add_color_brush_popup():
	var popup = preload("res://addons/m_terrain/gui/mtools_create_color_brush.tscn").instantiate()
	add_child(popup)
	popup.size = get_viewport_rect().size / 3
	
	var existing_brushes = color_brush_layer.layers.map(func(a): return a.NAME)
	if color_brush_name == "Color Paint":
		popup.init_for_color(existing_brushes)
	elif color_brush_name == "Channel Painter":
		popup.init_for_channel(existing_brushes)
	elif color_brush_name == "Bitwise":
		popup.init_for_bitwise(existing_brushes)
	elif color_brush_name == "Paint 16":
		popup.init_for_16(existing_brushes)
	elif color_brush_name == "Paint 256":
		popup.init_for_256(existing_brushes)
	popup.brush_created.connect(update_color_brush.bind(-1))
	
func update_color_brush(brush_name, brush_icon, data, id):	
	if id == -1:
		color_brush_layer.layers_num += 1
		id = color_brush_layer.layers_num -1
	color_brush_layer.layers[id].NAME = brush_name
	color_brush_layer.layers[id].ICON = brush_icon
	if "color" in data.keys():
		color_brush_layer.layers[id].color = data.color		
		color_brush_layer.layers[id].hardness = data.hardness
	elif "r_on" in data.keys():
		color_brush_layer.layers[id].red = data.r_on
		color_brush_layer.layers[id].green = data.g_on
		color_brush_layer.layers[id].blue = data.b_on
		color_brush_layer.layers[id].alpha = data.a_on
		color_brush_layer.layers[id]["red-value"] = data.r_value
		color_brush_layer.layers[id]["green-value"] = data.g_value
		color_brush_layer.layers[id]["blue-value"] = data.b_value
		color_brush_layer.layers[id]["alpha-value"] = data.a_value
		color_brush_layer.layers[id].hardness = data.hardness
	elif "bit" in data.keys():		
		color_brush_layer.layers[id].value = data.bit_value
		color_brush_layer.layers[id].bit = data.bit
	elif "paint16layer" in data.keys():
		color_brush_layer.layers[id]['paint-layer'] = data.paint16layer
	elif "paint256layer" in data.keys():
		color_brush_layer.layers[id]['paint-layer'] = data.paint256layer	
		
	init_color_brushes(active_terrain, color_layer_group_id)

func remove_color_brush(id):
	var group: MBrushLayers = active_terrain.brush_layers[color_layer_group_id]
	var brushes = group.layers.duplicate()	
	var index = 0
	if id > brushes.size()-1 or id < 0: 
		push_error("trying to delete nonexistant brush. Id: ", id)
		return
	for i in brushes.size():
		if i != id:			
			group.layers[index] = brushes[i]
			index += 1
	group.layers_num -= 1	
	
	init_color_brushes.call_deferred(active_terrain, color_layer_group_id)
#endregion

#region Grass Brushes
func init_grass_brushes():
	clear_brushes()
	brush_mode = &"grass"	
	var brush_item_scene = preload("res://addons/m_terrain/gui/mtools_brush_item.tscn")
	var brush_item = brush_item_scene.instantiate()
	brush_container.add_child(brush_item)		
	brush_item.set_text_brush("Add Grass")
	brush_item.brush_selected.connect(func(): on_grass_brush_select(0))
	
	brush_item = brush_item_scene.instantiate()
	brush_container.add_child(brush_item)		
	brush_item.set_text_brush("Remove Grass")	
	brush_item.brush_selected.connect(func(): on_grass_brush_select(1))
		
	on_grass_brush_select(0)

func on_grass_brush_select(id):
	icon = null
	if id==0:
		is_grass_add = true
		text = "Add Grass"		
	else:
		is_grass_add = false
		text = "Remove Grass"
	
#endregion


#region MNavigation Brushes
func init_mnavigation_brushes():
	clear_brushes()
	
	brush_mode = &"navigation"

	var brush_item_scene = preload("res://addons/m_terrain/gui/mtools_brush_item.tscn")
	var brush_item = brush_item_scene.instantiate()
	brush_container.add_child(brush_item)		
	brush_item.set_text_brush("Add Navigation")
	brush_item.brush_selected.connect(func(): on_grass_brush_select(0))
	
	brush_item = brush_item_scene.instantiate()
	brush_container.add_child(brush_item)		
	brush_item.set_text_brush("Remove Navigation")	
	brush_item.brush_selected.connect(func(): on_grass_brush_select(1))
	
	on_mnavigation_brush_select(0)

func on_mnavigation_brush_select(id):
	if id==0:
		is_grass_add = true
		text = "Add Navigation"
	else:
		is_grass_add = false
		text = "Remove Navigation"
	
#endregion

#region Input

func process_input(event):		
	if brush_mode == &"sculpt":				
		if event.keycode == KEY_SHIFT:			
			if event.is_pressed():
				if not event.echo:
					last_height_brush_id = height_brush_id
					height_brush_id = smooth_brush_id							
			else:				
				height_brush_id = last_height_brush_id
		elif event.keycode == KEY_CTRL:
			if event.is_pressed():
				if not event.echo:
					last_height_brush_id = height_brush_id
					height_brush_id = to_height_brush_id
			else:
				height_brush_id = last_height_brush_id
		elif event.keycode == KEY_ALT:			
			if not event.echo:					
				reverse_property_control.set_value(not reverse_property_control.value)
				
	elif brush_mode == &"paint":		
		pass
	elif brush_mode == &"grass": 		
		if event.keycode == KEY_ALT:
			if event.is_pressed():
				if not event.echo:
					is_grass_add = not is_grass_add
			else:
				is_grass_add = not is_grass_add	
#endregion


func _on_resized():	
	var vbox = get_child(0)
	var settings = find_child("brush_settings_panel")
	settings.custom_minimum_size.x = global_position.x-owner.global_position.x
	settings.size = settings.custom_minimum_size
	var size_panel = find_child("brush_size_panel")
	size_panel.custom_minimum_size.x = size.x
	size_panel.size = size_panel.custom_minimum_size
	var brushes_panel = find_child("brush_brushes_panel")
	brushes_panel.custom_minimum_size.x = (owner.size.x - size_panel.size.x - settings.size.x - 12) *0.5
	brushes_panel.size = brushes_panel.custom_minimum_size
	
	vbox.size.x = settings.size.x + size_panel.size.x + brushes_panel.size.x * 1.01
	if settings.visible:
		vbox.global_position.x = owner.global_position.x
	else:
		vbox.position.x = 0
		
	vbox.size.y = get_viewport_rect().size.y/5
	vbox.position.y = -vbox.size.y


func _on_panel_visibility_changed():	
	_on_resized()
