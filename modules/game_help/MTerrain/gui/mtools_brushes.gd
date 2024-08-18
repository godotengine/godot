#BRUSH button
@tool
extends Button

@onready var brush_container:ItemList = find_child("brush_container")
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

var color_layer_group_id:int
var color_brush_index:int
var color_brush_uniform:String
var color_brush_name:String

var no_image = preload("res://addons/m_terrain/icons/no_images.png") #For color brush

var property_element_list:Array
var color_brush_layers:Array
var color_brush_titles:Array
var selected_layer_group=0


func _ready():
	var panel = get_child(0)
	panel.visible = false
	panel.position.y = -panel.size.y
	panel.gui_input.connect(fix_gui_input)
	panel.size.x = get_viewport().size.x - global_position.x
	
func fix_gui_input(event: InputEvent):
	if event is InputEventMouseButton:
		get_viewport().set_input_as_handled()
	
func _toggled(toggled_on):
	if toggled_on:
		get_child(0).size.x = get_viewport().size.x - global_position.x
 
func clear_brushes():
	brush_container.clear()	
	clear_property_element()
	set("theme_override_styles/normal", null)
	text = ""
	for connection in brush_container.get_signal_connection_list("item_selected"):
		connection.signal.disconnect(connection.callable)	

#region Height Brushes
func init_height_brushes(new_brush_manager):		
	clear_brushes()	
	brush_mode = &"sculpt"	
	brush_container.item_selected.connect(on_height_brush_select)	
	height_brush_manager = new_brush_manager
	smooth_brush_id = height_brush_manager.get_height_brush_id("Smooth")
	raise_brush_id = height_brush_manager.get_height_brush_id("Raise")
	to_height_brush_id = height_brush_manager.get_height_brush_id("To Height")
	hole_brush_id = height_brush_manager.get_height_brush_id("Hole")

	var brush_names = height_brush_manager.get_height_brush_list()
	
	for n in brush_names:			
		var id = brush_container.add_item(n)
		if "raise" in n.to_lower():			
			brush_container.set_item_icon(id, preload("res://addons/m_terrain/icons/brush_icon_raise.svg"))
		if "height" in n.to_lower():
			brush_container.set_item_icon(id, preload("res://addons/m_terrain/icons/brush_icon_to_height.svg"))
		if "smooth" in n.to_lower():
			brush_container.set_item_icon(id, preload("res://addons/m_terrain/icons/brush_icon_smooth.svg"))
		if "hole" in n.to_lower():
			brush_container.set_item_icon(id, preload("res://addons/m_terrain/icons/brush_icon_hole.svg"))	
		if "layer" in n.to_lower():
			brush_container.set_item_icon(id, preload("res://addons/m_terrain/icons/eraser_icon.svg"))	
			
	on_height_brush_select(0)	
	
func on_height_brush_select(index):
	clear_property_element()
	if index < -1: return
	height_brush_id = index
	var brush_props = height_brush_manager.get_height_brush_property(height_brush_id)
	for p in brush_props:
		create_props(p)
	text = ""# brush_container.get_item_text(index)
	set("theme_override_styles/normal", null)
	icon = brush_container.get_item_icon(index)	
	tooltip_text = "Current brush: " + brush_container.get_item_text(index)

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
		
	brush_container.clear()	
	var layer_group = active_terrain.get_layers_info()[layer_group_id]						
	
	var index = -1
	for i in layer_group["info"]:				
		index +=1
		var layer_name:String= i["name"]
		if layer_name.is_empty():
			layer_name = "layer "+str(index)
		var icon:Texture = i["icon"]
		if not icon:
			icon = no_image
		var id = brush_container.add_item(layer_name,icon)
		brush_container.set_item_custom_bg_color(index,i["icon-color"])
	color_layer_group_id = layer_group["index"]
	color_brush_uniform = layer_group["uniform"]
	color_brush_name = layer_group["brush_name"]
	brush_container.item_selected.connect(
		func(id):
			brush_layer_selected(id, layer_group)
	)
	if brush_container.item_count !=0:
		brush_layer_selected(0, 0)

func brush_layer_selected(index, layer_group):			
	active_terrain.set_color_layer(index, color_layer_group_id,color_brush_name)
	var brush_icon = brush_container.get_item_icon(index)
	if brush_icon and brush_icon != no_image:
		icon = brush_icon
	else:
		icon = null
	text = "" #brush_container.get_item_text(index)
	tooltip_text = "Current brush: " + brush_container.get_item_text(index)
	var color = brush_container.get_item_custom_bg_color(index)
	if color:
		var stylebox = StyleBoxFlat.new()
		stylebox.bg_color = color
		set("theme_override_styles/normal", stylebox)
	else:
		set("theme_override_styles/normal", null)

#endregion

#region Grass Brushes
func init_grass_brushes():
	clear_brushes()
	
	brush_mode = &"grass"	
	brush_container.item_selected.connect(on_grass_brush_select)
		
	brush_container.add_item("Add Grass")
	brush_container.add_item("Remove Grass")
	brush_container.select(0)
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
	brush_container.item_selected.connect(on_mnavigation_brush_select)
		
	brush_container.add_item("Add Navigation")
	brush_container.add_item("Remove Navigation")
	brush_container.select(0)

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
		#to do: add remove paint?
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
	var size_panel = find_child("brush_size_panel")
	size_panel.custom_minimum_size.x = size.x
	var brushes_panel = find_child("brush_brushes_panel")
	brushes_panel.custom_minimum_size.x = owner.size.x - size_panel.size.x - settings.size.x - 12
	
	vbox.size.x = owner.size.x
	vbox.global_position.x = owner.global_position.x
	vbox.size.y = get_viewport_rect().size.y/5
	vbox.position.y = -vbox.size.y


func _on_panel_visibility_changed():
	_on_resized()
