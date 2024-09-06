@tool
extends Button

signal edit_mode_changed

var item_container
var exit_edit_mode_button 
var edit_selected_button
var active_object

var more_options_icon = preload("res://addons/m_terrain/icons/more_options_icon.svg")

var sculpt_icon
var paint_icon
var grass_icon
var path_icon
var curve_mesh_icon
var navigation_icon

func _ready():
	var panel = get_child(0)
	panel.visible = false
	panel.position.y = -panel.size.y-1
	
	item_container = find_child("edit_mode_item_container")
	exit_edit_mode_button = get_node("../edit_mode_exit_button")		
	edit_selected_button = get_node("../edit_selected_button")		
	
	if not edit_selected_button.pressed.is_connected(edit_selected):
		edit_selected_button.pressed.connect(edit_selected)
	edit_selected_button.visible = false
	
	sculpt_icon = load("res://addons/m_terrain/icons/foilage_icon.png")
	paint_icon = load("res://addons/m_terrain/icons/brush_icon.png")
	grass_icon = load("res://addons/m_terrain/icons/foilage_icon.png")
	path_icon = load("res://addons/m_terrain/icons/MPath.svg")
	curve_mesh_icon = load("res://addons/m_terrain/icons/MCurveMesh.svg")
	navigation_icon = load("res://addons/m_terrain/icons/MNavigationRegion3D.png")
	
	if not exit_edit_mode_button.pressed.is_connected(exit_edit_mode_button_pressed):
		exit_edit_mode_button.pressed.connect(exit_edit_mode_button_pressed)
	if not exit_edit_mode_button.pressed.is_connected(exit_edit_mode_button.hide):	
		exit_edit_mode_button.pressed.connect(exit_edit_mode_button.hide)

func init_edit_mode_options(all_mterrain):	
	var button_template =  Button.new()
	button_template.mouse_filter = Control.MOUSE_FILTER_PASS
	button_template.alignment = HORIZONTAL_ALIGNMENT_LEFT
	var biggest_button_size = 0
	for child in item_container.get_children():
		child.queue_free()
	
	if all_mterrain.size() != 0:
		for terrain in all_mterrain:
			var button = button_template.duplicate()
			button.icon = sculpt_icon
			button.text = "Sculpt " + terrain.name
			item_container.add_child(button)
			biggest_button_size = max(biggest_button_size, button.size.x)
			button.pressed.connect(edit_selected.bind(terrain))		
			
			button = button_template.duplicate()
			button.text = "Paint " + terrain.name
			button.icon = paint_icon
			item_container.add_child(button)
			button.pressed.connect(edit_selected.bind(terrain, &"paint"))
	var all_grass = MGrass.get_all_grass_nodes()
	for child in all_grass:
		var button:Button = button_template.duplicate()
		button.text = "Paint " + child.name
		button.icon = grass_icon
		item_container.add_child(button)
		biggest_button_size = max(biggest_button_size, button.size.x)
		button.pressed.connect(edit_selected.bind(child))
	var all_navigation = MNavigationRegion3D.get_all_navigation_nodes()
	for child in all_navigation:
		var button = button_template.duplicate()
		button.text = "Paint " + child.name
		button.icon = navigation_icon
		item_container.add_child(button)
		biggest_button_size = max(biggest_button_size, button.size.x)
		button.pressed.connect(edit_selected.bind(child))
	var all_path = MPath.get_all_path_nodes()
	var all_curve_mesh = MCurveMesh.get_all_curve_mesh_nodes()
	for child in all_path:
		var button = button_template.duplicate()
		button.text = "Edit " + child.name
		button.icon = path_icon
		item_container.add_child(button)
		biggest_button_size = max(biggest_button_size, button.size.x)
		button.pressed.connect(edit_selected.bind(child))
	for child in all_curve_mesh:
		var button = button_template.duplicate()
		button.text = "Edit " + child.name
		button.icon = curve_mesh_icon
		item_container.add_child(button)
		biggest_button_size = max(biggest_button_size, button.size.x)
		button.pressed.connect(edit_selected.bind(child))
	button_template.queue_free()
	get_child(0).size. x = biggest_button_size + 12	
	
func change_active_object(object):	
	#In future, make it auto-switch to the same edit mode, just for different object
	if not object == active_object:
		exit_edit_mode_button_pressed()
		active_object = null
	exit_edit_mode_button.visible = false
	edit_selected_button.visible = true
	if object is MTerrain:
		active_object = object
		edit_selected_button.text = "Click to Sculpt " + object.name
	elif object is MGrass or object is MNavigationRegion3D:
		active_object = object
		edit_selected_button.text = "Click to Paint " + object.name			
	elif object is MPath or object is MCurveMesh:
		active_object = object
		edit_selected_button.text = "Click to Edit " + object.name		
	else:
		edit_selected_button.visible = false
		
	text = ""
	theme_type_variation
	icon = more_options_icon
	
func exit_edit_mode_button_pressed():	
	edit_mode_changed.emit(null, &"")
	change_active_object(active_object)
	


func edit_selected(object = active_object, override_mode=null):
	if object is MTerrain:
		if override_mode and override_mode==&"paint":
			text = "Paint " + object.name
			theme_type_variation = ""
			icon = null
			edit_mode_changed.emit(object, &"paint")		
		else:
			text = "Sculpt " + object.name
			theme_type_variation = ""
			icon = null
			edit_mode_changed.emit(object, &"sculpt")		
		edit_selected_button.visible = false			
		exit_edit_mode_button.show()		
		active_object = object
	elif object is MGrass or object is MNavigationRegion3D:
		text = "Paint " + object.name
		theme_type_variation = ""
		icon = null
		edit_mode_changed.emit(object, &"paint")
		edit_selected_button.visible = false
		exit_edit_mode_button.visible = true
		active_object = object				
	elif object is MPath:
		text = "Edit " + object.name
		theme_type_variation = ""
		icon = null
		edit_mode_changed.emit(object, &"mpath")
		edit_selected_button.visible = false
		exit_edit_mode_button.show()		
		active_object = object
	elif object is MCurveMesh:
		text = "Edit " + object.name
		theme_type_variation = ""
		icon = null
		edit_mode_changed.emit(object, &"mcurve_mesh")
		edit_selected_button.visible = false
		exit_edit_mode_button.show()		
		active_object = object
