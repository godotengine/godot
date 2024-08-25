@tool
extends Button

signal layer_changed

@onready var layers_container:Control = find_child("layer_item_container")
@onready var add_height_layer_button: Button = find_child("add_height_layer_button")
@onready var add_color_layer_button: Button = find_child("add_color_layer_button")
@onready var merge_height_layers_button: Button = find_child("merge_height_layers_button")
@onready var height_layer_controls: Control = find_child("layer_controls")
var active_terrain: MTerrain = null
var active_heightmap_layer = ""

var hide_icon = preload("res://addons/m_terrain/icons/hidden.png")
var show_icon = preload("res://addons/m_terrain/icons/show.png")
var layer_item_scene = preload("res://addons/m_terrain/gui/mtools_layer_item.tscn")
var stylebox_selected = preload("res://addons/m_terrain/gui/styles/stylebox_selected.tres")

var brush_control: Control
var confirmation_popup_scene = preload("res://addons/m_terrain/gui/mtools_layer_warning_popup.tscn")


func _ready():
	var panel:Control = get_child(0)
	panel.visible = false
	panel.position.y = -panel.size.y-2
	panel.gui_input.connect(fix_gui_input)
	panel.visibility_changed.connect(func():
		var max_width = 0
		for child in layers_container.get_children():
			if child.get_total_width() > max_width:
				max_width = child.get_total_width()
		max_width = min(max_width, owner.mtools_root.get_child(0).size.x)		
		panel.custom_minimum_size.x = max_width
		panel.size.x = max_width
	)
	
	merge_height_layers_button.pressed.connect(merge_all_heightmap_layers)
	add_height_layer_button.pressed.connect(add_heightmap_layer)
	add_color_layer_button.pressed.connect(add_color_layer)

func fix_gui_input(event: InputEvent):
	if event is InputEventMouseButton:
		get_viewport().set_input_as_handled()

func clear_layers():
	add_height_layer_button.visible = false
	merge_height_layers_button.visible = false
	add_color_layer_button.visible = false		
	for child in layers_container.get_children():
		child.queue_free()
		layers_container.remove_child(child)
		
#region Height Layers
func init_height_layers(mterrain:MTerrain):
	active_terrain = mterrain
	clear_layers()
	add_height_layer_button.visible = true
	merge_height_layers_button.visible = true
	for layer in active_terrain.heightmap_layers:	
		var layer_item = layer_item_scene.instantiate()
		layer_item.name = layer		
		layers_container.add_child(layer_item)
		layer_item.custom_minimum_size = custom_minimum_size
		layer_item.init_for_heightmap()
		layer_item.layer_selected.connect(change_heightmap_layer_selection)
		layer_item.layer_visibility_changed.connect(toggle_heightmap_layer_visibility)
		layer_item.layer_renamed.connect(rename_heightmap_layer)
		#layer_item.layer_index_changed.connect(????)
		layer_item.layer_removed.connect(remove_heightmap_layer)	
		layer_item.layer_merged_with_background.connect(merge_heightmap_layer_with_background)
	if layers_container.get_child_count()!=0:
		layers_container.get_child(0).select_layer.call_deferred()


func add_heightmap_layer():
	var i = 0
	while str("New Layer ", i) in active_terrain.heightmap_layers:
		i+= 1
	active_terrain.add_heightmap_layer(str("New Layer ", i))
	init_height_layers(active_terrain)	

func remove_heightmap_layer(layer_name):
	active_terrain.set_active_layer_by_name(layer_name)
	active_terrain.remove_heightmap_layer()
	if layer_name == active_heightmap_layer:
		change_heightmap_layer_selection(0, "background")			
	else:
		active_terrain.set_active_layer_by_name(active_heightmap_layer)

func toggle_heightmap_layer_visibility(layer_name):		
	active_terrain.set_active_layer_by_name(layer_name)
	active_terrain.toggle_heightmap_layer_visibile()
	active_terrain.set_active_layer_by_name(active_heightmap_layer)

func change_heightmap_layer_selection(id, layer_name):
	active_heightmap_layer = layer_name
	active_terrain.set_active_layer_by_name(layer_name)
	for layer in layers_container.get_children():
		if layer.name == layer_name:
			layer.selected = true
			layer.set("theme_override_styles/panel", stylebox_selected)
		else:			
			layer.selected = false
			layer.set("theme_override_styles/panel", StyleBoxEmpty.new())
	text = layer_name	
	layer_changed.emit(layer_name)

func merge_all_heightmap_layers():
	var popup = confirmation_popup_scene.instantiate()
	add_child(popup)
	popup.confirmed.connect( func():	
		for layer in active_terrain.heightmap_layers:
			if layer == "background": continue		
			active_terrain.set_active_layer_by_name(layer)
			active_terrain.merge_heightmap_layer()
		init_height_layers(active_terrain)
	)
		
func merge_heightmap_layer_with_background(layer):			
	active_terrain.set_active_layer_by_name(layer)
	active_terrain.merge_heightmap_layer()	

func rename_heightmap_layer(name_button, new_name):
	if new_name == "": return
	if new_name in active_terrain.heightmap_layers: return
	var layers = active_terrain.heightmap_layers
	
	for i in layers.size():
		if layers[i] == name_button.text:
			if active_terrain.rename_heightmap_layer(layers[i], new_name):
				layers[i] = new_name
		
	name_button.text = new_name
	name_button.name = new_name
	text = new_name
	
	#init_height_layers(active_terrain)
	
#endregion

#region Color Layers
func init_color_layers(mterrain:MTerrain = active_terrain, brush_button = brush_control):
	if not mterrain or not brush_button:
		push_error("init color layers failed: no mterrain or no brush button" )
		return
	active_terrain = mterrain
	clear_layers()
	add_color_layer_button.visible = true
	brush_control = brush_button
	var valid_count:int = 0
	for i in range(active_terrain.brush_layers.size()):
		if validate_brush_layer_data(active_terrain.brush_layers[i]):
			add_color_layer_item(active_terrain, i)
			valid_count += 1
		else:
			active_terrain.brush_layers.erase(active_terrain.brush_layers[i])
			active_terrain.brush_layers_groups_num -= 1
	if valid_count > 0:
		layers_container.get_child(0).select_layer()
	else:
		text = "(click to add a color layer)"
		change_color_layer_selection(-1, "")

func validate_brush_layer_data(layer):
	if layer.uniform_name in active_terrain.get_image_list():
		return true
	else:
		
		push_error("layer ", layer.layers_title, " has invalid uniform: ", layer.uniform_name)		
	
func add_color_layer_item(terrain:MTerrain, layer_group_index:int):
	var layer_item = layer_item_scene.instantiate()
	#layer_item.name = layer.layers_title if layer.layers_title != "" else str("layer group ", layer_group_id)
	layers_container.add_child(layer_item)
	layer_item.init_for_colors(terrain,layer_group_index)		
	#layer_item.name_button.tooltip_text = str("select ", layer.layers_title, ". Uniform: ", layer.uniform_name)
	layer_item.layer_selected.connect(change_color_layer_selection)
	layer_item.color_layer_removed.connect(remove_color_layer)
	
func change_color_layer_selection(layer_id, layer_name):		
	if layer_id < 0:			
		layer_changed.emit(layer_id)		
	else:		
		brush_control.init_color_brushes(active_terrain, layer_id)
		text = layer_name	
		layer_changed.emit(layer_id)
			

func add_color_layer():
	var window = preload("res://addons/m_terrain/gui/color_layer_creator_window.tscn").instantiate()
	add_child(window)
	window.set_terrain(active_terrain)
	window.layer_created.connect(func(layer): 
		add_color_layer_item(active_terrain, active_terrain.brush_layers_groups_num-1 ) 
		change_color_layer_selection(active_terrain.brush_layers_groups_num-1, active_terrain.brush_layers[active_terrain.brush_layers_groups_num-1].layers_title)
	)

func rename_color_layer():
	pass
	
func remove_color_layer(terrain:MTerrain, layer_group_index:int,both:bool):
	var layer:MBrushLayers = terrain.brush_layers[layer_group_index]
	var dname = layer.uniform_name
	if both:
		owner.remove_image(active_terrain, dname)
	var layers = terrain.brush_layers
	var id = 0
	var remove_count = 0
	if both:
		var i:int=terrain.brush_layers.size() - 1
		while  i >= 0:
			if(terrain.brush_layers[i].uniform_name == dname):
				terrain.brush_layers.remove_at(i)
			i-=1
	else:
		terrain.brush_layers.remove_at(layer_group_index)
	terrain.brush_layers_groups_num = terrain.brush_layers.size()
	if 	terrain.brush_layers.size() > 0:
		change_color_layer_selection(0, terrain.brush_layers[0].layers_title)
	else:
		change_color_layer_selection(-1, "")
	init_color_layers()

#endregion

func init_grass_toggle():
	pass
