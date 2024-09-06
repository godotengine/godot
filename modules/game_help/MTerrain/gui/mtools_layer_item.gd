@tool
extends Control

signal layer_visibility_changed
signal layer_renamed
signal layer_index_changed
signal layer_removed
signal layer_selected
signal color_layer_removed
signal layer_merged_with_background


@onready var visibility_button = $hbox/visible
@onready var name_button = $hbox/name
@onready var merge_down_button = $hbox/merge_down
@onready var remove_button = $hbox/remove
@onready var move_up_button = $hbox/move_up
@onready var move_down_button = $hbox/move_down
@onready var rename_button = $hbox/rename
@onready var rename_input = $hbox/rename_input
var icon_visible = preload("res://addons/m_terrain/icons/eye.svg")
var icon_hidden = preload("res://addons/m_terrain/icons/eye-close.svg")

var selected = false

var confirmation_popup_scene = preload("res://addons/m_terrain/gui/mtools_layer_warning_popup.tscn")

func _ready():
	name_button.pressed.connect(select_layer)	

func disconnect_signals():			
#	for connection in name_button.get_signal_connection_list("pressed"):
#		connection.signal.disconnect(connection.callable)	 
#	for connection in remove_button.get_signal_connection_list("pressed"):
#		connection.signal.disconnect(connection.callable)	 
	for connection in rename_button.get_signal_connection_list("pressed"):
		connection.signal.disconnect(connection.callable)	 
	for connection in rename_input.get_signal_connection_list("text_submitted"):
		connection.signal.disconnect(connection.callable)	 
	for connection in rename_input.get_signal_connection_list("focus_exited"):
		connection.signal.disconnect(connection.callable)	 

func select_layer():
	layer_selected.emit(get_index(), name_button.text)

func remove_heightmap_layer():
	var popup = confirmation_popup_scene.instantiate()
	add_child(popup)
	popup.confirmed.connect( func():
		layer_removed.emit(name)		
		queue_free()
	)


func init_for_heightmap():
	disconnect_signals()
	if name == "background": 
		rename_button.queue_free()
		remove_button.queue_free()
		move_up_button.queue_free()
		move_down_button.queue_free()
		visibility_button.queue_free()			
	name_button.text = name
	visibility_button.toggled.connect(change_visibility)	
	merge_down_button.pressed.connect(merge_layer_down)
	rename_button.pressed.connect(begin_rename)
	rename_input.text_submitted.connect(end_rename)
	rename_input.focus_exited.connect(end_rename.bind(""))
	remove_button.pressed.connect(remove_heightmap_layer)
	
func change_visibility(toggle_on):
	visibility_button.icon = icon_hidden if toggle_on else icon_visible
	layer_visibility_changed.emit(name)

func merge_layer_down():
	var popup = confirmation_popup_scene.instantiate()
	add_child(popup)
	popup.confirmed.connect( func():
		layer_merged_with_background.emit(name)
		queue_free()
	)

func begin_rename():
	rename_input.text = name_button.text
	rename_input.visible = true
	name_button.visible = false
	rename_button.visible = false
	rename_input.grab_focus()
	
func end_rename(_new_name=""):
	if rename_input.text == "": return
	rename_input.visible = false
	name_button.visible = true
	rename_button.visible = true
	layer_renamed.emit(name_button, rename_input.text)
	

func remove_color_layer(terrain:MTerrain,layer_group_index:int):	
	var popup = preload("res://addons/m_terrain/gui/mtools_popup_remove_color_layer.tscn").instantiate()
	var layer:MBrushLayers = terrain.brush_layers[layer_group_index]
	add_child(popup)
	var uniform = layer.uniform_name
	var layers_with_same_uniform = []
	for l in terrain.brush_layers:
		if l.uniform_name == uniform and l.layers_title != name:
			layers_with_same_uniform.push_back(l.layers_title)
	popup.set_shared_uniform_label(layers_with_same_uniform)
	popup.confirmed.connect( func(both):
		color_layer_removed.emit(terrain, layer_group_index,both)
		queue_free()
	)

func init_for_colors(terrain:MTerrain, layer_group_index:int):	
	var layer:MBrushLayers = terrain.brush_layers[layer_group_index]
	disconnect_signals()
	rename_button.queue_free()
	rename_input.queue_free()
	#remove_button.queue_free()
	move_up_button.queue_free()
	move_down_button.queue_free()
	merge_down_button.queue_free()
	visibility_button.queue_free()		

	name_button.text = layer.layers_title if layer.layers_title != "" else str("layer group ", layer_group_index)
	remove_button.pressed.connect(remove_color_layer.bind(terrain,layer_group_index))
	#rename_button.pressed.connect(begin_rename)
	#rename_input.text_submitted.connect(end_rename)
	#rename_input.focus_exited.connect(end_rename.bind(""))
	
	
#region Theme: color and size etc
func _on_resized():
	resize_children_recursive(self, custom_minimum_size.y)

func resize_children_recursive(parent, new_size):
	for child in parent.get_children():
		if child is Control:
			child.custom_minimum_size.x = new_size
			child.custom_minimum_size.y = new_size		
		resize_children_recursive(child, new_size)
		
func get_total_width():	
	var total = name_button.text.length() * get_theme_default_font_size() *0.75	
	#name_button.size.x if is_instance_valid(name_button) and name_button.visible else total
	
	total = total + visibility_button.size.x if is_instance_valid(visibility_button) and visibility_button.visible else total	
	total = total + rename_button.size.x if is_instance_valid(rename_button) and rename_button.visible else total
	total = total + merge_down_button.size.x if is_instance_valid(merge_down_button) and merge_down_button.visible else total
	total = total + remove_button.size.x if is_instance_valid(remove_button) and remove_button.visible else total
	total *= 1.01
	return total
#endregion
