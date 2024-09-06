@tool
extends Control

##############################################################
# 							MTools
#	1. m_terrain.gd sets mask, decal, and human on enter_tree()
#   2. User selects a node
#	   This calls edit_mode_button.init_edit_mode_options()
#      This populates the list of possible edit modes.	   
#	3. User selectes edit mode by clicking on edit_mode_button,
#	   This calls mtools.set_edit_mode()
#	   This sets the active_object and current_mode  
#   4. This sets appropriate "layers" (heightmap or paint) 
#	   This sets appropriate "brushes" (height, color, grass, nav)
#      If mpath, show mpath menu
#	   If mcurve_mesh, show mcurve_mesh menu
#   5. m_terrain.gd _forward_gui_input() calls the draw() when needed
#
#	NOTE: "popup" panels are just panels. 
#   They appear on button click and we use mouse_exit to hide them. 
#   We use timer to add a delay to prevent accidents
#
#   The popup buttons are also the main scripts for that function
#   e.g. layers_popup_button has all the code to do with layers
#	e.g. brush_popup_button has all the code to do with brushes
#
###############################################################

# fix icon-only button margins
# fix brush bool-property button 
# added ctrl mouse wheel to increase brush size
# fixed xyz lock not working from buttons... 
# right click cancel drag for mpath 
# grass auto make active when entering paint mode
# has_sublayer merge sublayer 

# terrain walk mode

#region Signals
signal request_save
signal request_info_window
signal request_import_window
signal request_image_creator
signal edit_mode_changed
#endregion

#region UI Controls
@onready var status_bar: Control = find_child("status_bar")
#@onready var save_button: Control = find_child("save_button") #TO DELETE
@onready var paint_panel: Control = find_child("paint_panel")

@onready var options_popup_button: Button = find_child("options_button")

@onready var edit_mode_button: Button = find_child("edit_mode_button")
@onready var layers_popup_button: Button = find_child("layers_button")
@onready var brush_popup_button: Button = find_child("brush_button")
@onready var mask_popup_button: Button = find_child("mask_button")

var active_snap_object:Node3D = null

var collision_ray_step=0.2
var ray_col:MCollision
var col_dis:float

var undo_redo: EditorUndoRedoManager
@onready var mpath_gizmo_gui = find_child("mpath_gizmo_gui")
@onready var mcurve_mesh_gui = find_child("mcurve_mesh")
@onready var brush_size_control: Control = find_child("brush_size")

@onready var mtools_root = find_child("mtools_root")

@onready var popup_buttons = [
	edit_mode_button,
	options_popup_button,
	layers_popup_button,
	brush_popup_button, 	
	mask_popup_button,
]

var current_popup_button: Button = null
var timer
#endregion

var current_edit_mode = &"" # &"", &"sculpt", &"paint"
var active_object = null



var brush_manager:MBrushManager = MBrushManager.new()
var brush_decal # set by m_terrain.gd on enter_tree()
var mask_decal # set by m_terrain.gd on enter_tree()
var human_male # set by m_terrain.gd on enter_tree()
var edit_human_position = false
@onready var human_button = find_child("human_male")
@onready var grass_merge_sublayer_button = find_child("grass_merge_sublayer")
@onready var walk_terrain_button = find_child("walk_terrain")
var walking_terrain = false
var editor_camera: Camera3D = null #this is set by forward_gui_input
var walk_speed = 5 #in meters per seccond

#region Initialisations
func _ready():	
	timer = Timer.new()
	timer.timeout.connect(func(): current_popup_button.button_pressed = false)
	add_child(timer)
	update_theme()
	for button in popup_buttons:
		init_popup_button_signals(button)	

	edit_mode_button.edit_mode_changed.connect(set_edit_mode)		

	theme_changed.connect(update_theme)
	visibility_changed.connect(_on_resized)
	
	MTool.enable_editor_plugin()

func set_brush_decal(new_brush_decal):
	brush_decal = new_brush_decal
	brush_decal.visible = false
	brush_size_control.value_changed.connect(brush_decal.set_brush_size)			

func set_mask_decal(new_mask):
	mask_decal = new_mask
	mask_decal.visible = false
	mask_popup_button.init_masks(mask_decal,  find_child("mask_size"), find_child("mask_rotation"),find_child("mask_cutoff"), find_child("invert_mask_button"))			

func on_node_modified(node):	
	if node is MTerrain or node is MGrass or node is MNavigationRegion3D or node is MPath or node is MCurveMesh:
		update_edit_mode_options()
		
func update_edit_mode_options():	
	var all_mterrain = MTerrain.get_all_terrain_nodes()
	edit_mode_button.init_edit_mode_options(all_mterrain)

func clear_current_popup_button():
	if current_popup_button:
		current_popup_button.button_pressed = false			

func init_popup_button_signals(popup_button:Button):	
	popup_button.button_pressed = false
	popup_button.toggled.connect(
		func(toggled_on): 
			if toggled_on: 	
				clear_current_popup_button()
				popup_button.get_child(0).visible = true			
				current_popup_button = popup_button
				if not popup_button.mouse_entered.is_connected(_on_mouse_entered_popup):
					popup_button.mouse_entered.connect(_on_mouse_entered_popup)				
			else:
				popup_button.get_child(0).visible = false
				current_popup_button = null
				if not popup_button.mouse_exited.is_connected(_on_mouse_exited_popup):
					popup_button.mouse_exited.connect(_on_mouse_exited_popup)				
	)

func _on_mouse_exited_popup() -> void:
	if current_popup_button:
		timer.one_shot = true	
		timer.start(0.35)

func _on_mouse_entered_popup() -> void:
	if is_instance_valid(timer):
		timer.stop()

#endregion

#region getters
func get_active_mterrain():
	var object = active_object
	if not object:
		var selection = EditorInterface.get_selection().get_selected_nodes()
		if selection.size() == 1:
			object = selection[0]
	if object is MTerrain:
		return object
	if object is MGrass or object is MNavigationRegion3D or object:
		if object.get_parent() is MTerrain:
			return object.get_parent()	
	#This is only used for snapping to MTerrain in MPath:
	if object is MPath or object is MCurveMesh:
		var all_mterrain = MTerrain.get_all_terrain_nodes()
		if all_mterrain.size()>0:
			return all_mterrain[0]
	
#func get_all_mgrass():
	#return MGrass.get_all_grass_nodes()
#
#func get_all_mnavigation(root):
	#return MNavigationRegion3D.get_all_navigation_nodes()
	#
#func get_all_mpath(root):
	#return MPath.get_all_path_nodes()

#endregion

func set_active_object(object):	
	if active_object == object: return	
	
	#Cleanup active object stuff before setting new active object
	if active_object is MNavigationRegion3D:
		active_object.set_npoints_visible(false)
		
	edit_mode_button.change_active_object(object)	
	#Automatically enter edit mode for mpath
	if object is MPath:
		edit_mode_button.edit_selected(object)

#region EditorPlugin functions
func process_input(event):
	if current_edit_mode in [&"sculpt", &"paint"]:
		if event is InputEventMouseButton and Input.is_key_pressed(KEY_CTRL):
			if event.button_index == MOUSE_BUTTON_WHEEL_UP:
				brush_decal.set_brush_size(brush_decal.get_brush_size() *1.05)
				brush_size_control.update_value(brush_decal.get_brush_size())
				return true
			elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				brush_decal.set_brush_size(brush_decal.get_brush_size() /1.05)
				brush_size_control.update_value(brush_decal.get_brush_size())
				return true
		if event is InputEventKey:			
			brush_popup_button.process_input(event)
			var paint_brush_resize_speed:float=1.0
			var paint_mask_resize_speed:float=1.0
			const max_paint_brush_resize_speed:float=8.0
			const max_mask_brush_resize_speed:float=16.0		
			
			if Input.is_action_just_released("mterrain_mask_size_increase") or Input.is_action_just_released("mterrain_mask_size_decrease"):
				paint_mask_resize_speed=1
			if Input.is_action_just_released("mterrain_brush_size_increase") or Input.is_action_just_released("mterrain_brush_size_decrease"):
				paint_brush_resize_speed=1				
				
			#if event.keycode == KEY_BRACKETLEFT or event.keycode == KEY_BRACKETRIGHT:
			if Input.is_action_pressed("mterrain_brush_size_increase"):				
				paint_brush_resize_speed = min(paint_brush_resize_speed+0.1,max_paint_brush_resize_speed)
				brush_decal.set_brush_size(brush_decal.get_brush_size() + floor(paint_brush_resize_speed))
				brush_size_control.update_value(brush_decal.get_brush_size())
				return true					
			#elif event.keycode == KEY_MINUS:
			elif Input.is_action_pressed("mterrain_brush_size_decrease"):				
				paint_brush_resize_speed = min(paint_brush_resize_speed+0.1,max_paint_brush_resize_speed)
				brush_decal.set_brush_size(brush_decal.get_brush_size() - floor(paint_brush_resize_speed))
				brush_size_control.update_value(brush_decal.get_brush_size())
				return true					
			#elif event.keycode == KEY_PERIOD:
			elif Input.is_action_pressed("mterrain_mask_size_increase"):				
				paint_mask_resize_speed = min(paint_mask_resize_speed+0.2,max_mask_brush_resize_speed)
				mask_decal.increase_size(paint_mask_resize_speed)
			#elif event.keycode == KEY_COMMA:
			elif Input.is_action_just_pressed("mterrain_mask_size_decrease"):
				if event.is_pressed():
					paint_mask_resize_speed = min(paint_mask_resize_speed+0.2,max_mask_brush_resize_speed)
					mask_decal.increase_size(-paint_mask_resize_speed)							
			#elif event.keycode == KEY_L and event.pressed:
			elif Input.is_action_just_pressed("mterrain_mask_rotate_clockwise"):
				mask_decal.rotate_image(1)
			#elif event.keycode == KEY_K and event.pressed:
			elif Input.is_action_just_pressed("mterrain_mask_rotate_counter_clockwise"):
				mask_decal.rotate_image(-1)				
			#elif event.keycode == KEY_SEMICOLON and event.pressed:
			elif Input.is_action_just_pressed("mterrain_mask_rotation_reset"):
				mask_decal.reset_image_rotation()
	if active_object is MGrass:
		status_bar.set_grass_label(active_object.get_count())
	else:
		status_bar.disable_grass_label()

#func update_brushes_based_on_heightmap_layer(new_layer):
	#if new_layer == "holes":	
		#brush_list_option.select(hole_brush_id)		
	#elif brush_id == hole_brush_id:
		#brush_list_option.select(raise_brush_id)		

func process_input_terrain_walk(cam:Camera3D, event:InputEvent):
	if event is InputEventMouseButton:
		if event.button_index in [MOUSE_BUTTON_MASK_RIGHT, MOUSE_BUTTON_MASK_LEFT]:
			walk_terrain_button.button_pressed = false
			return true
		if event.button_index == MOUSE_BUTTON_WHEEL_UP: 
			walk_speed *= 1.03
			return true
		if event.button_index == MOUSE_BUTTON_WHEEL_DOWN: 
			walk_speed /= 1.03
			return true
	if event is InputEventKey and event.keycode == KEY_ESCAPE:
		walk_terrain_button.button_pressed = false
		return true
	if event is InputEventMouseMotion:
		cam.rotation.y += -event.relative.x/100
		cam.rotation.x = clamp(cam.rotation.x -event.relative.y/100, -PI*0.3, PI*0.35)
		return true

func _process(delta):
	if walking_terrain and editor_camera:
		var multiplier = 1
		if Input.is_key_pressed(KEY_SHIFT):
			multiplier = 2
		editor_camera.position += editor_camera.basis.z * Input.get_axis("mterrain_walk_forward", "mterrain_walk_backward") * walk_speed * delta * multiplier
		editor_camera.position += editor_camera.basis.x * Input.get_axis("mterrain_walk_left", "mterrain_walk_right") * walk_speed * delta * multiplier
		editor_camera.position.y = get_active_mterrain().get_height(editor_camera.global_position) + 1.6

func on_selection_changed(obj):
	if obj is MTerrain:
		request_show()	
		return
	if obj.get_parent() is MTerrain:
		if obj is MGrass or obj is MNavigationRegion3D:
			request_show()	
			return
	if obj is MPath or obj is MCurveMesh:
		request_show()	
		return
	
	if mcurve_mesh_gui.obj and mcurve_mesh_gui.is_active():
		request_show()	
		return
	request_hide()

func on_handles(object):
	active_snap_object = null
	if mcurve_mesh_gui and mcurve_mesh_gui.obj and mcurve_mesh_gui.is_active(): 
		return false
	if mpath_gizmo_gui and not object is MPath:
		mpath_gizmo_gui.visible = false
	if not object is MCurveMesh:
		mcurve_mesh_gui.set_curve_mesh(null)	
		
	if object is MPath or (object is MCurveMesh and object.get_parent() is MPath) or object is MTerrain or (object is MGrass and object.get_parent() is MTerrain) or (object is MNavigationRegion3D and object.get_parent() is MTerrain):	
		set_active_object(object)
		request_show()				
		return true
	#for some reason these get selected when switching from grass paint mode to terrain sculpt/paint mode:
	if object is MTerrainMaterial or object is MBrushLayers:
		return false
	request_hide()		
	if object is Node3D:
		var all_terrain = MTerrain.get_all_terrain_nodes()
		for mterrain:MTerrain in all_terrain:
			if mterrain.is_grid_created():
				active_snap_object = object
				return false

func forward_3d_gui_input(viewport_camera, event):
	var active_terrain = get_active_mterrain()
	if not active_terrain is MTerrain: 
		ray_col = null
	elif event is InputEventMouse:
		var ray:Vector3 = viewport_camera.project_ray_normal(event.position)
		var pos:Vector3 = viewport_camera.global_position
		ray_col = active_terrain.get_ray_collision_point(pos,ray,collision_ray_step,1000)
	
	if walking_terrain:
		editor_camera = viewport_camera
		if process_input_terrain_walk(viewport_camera, event):
			return true
	
	### we don't need this any more as we grab editor camera in gdextension automaticly
	#for terrain in get_all_mterrain():
	#	terrain.set_editor_camera(viewport_camera)
	######################## HANDLE CURVE GIZMO ##############################
	if mpath_gizmo_gui.visible:				
		return mpath_gizmo_gui.gizmo._forward_3d_gui_input(viewport_camera, event, ray_col)
	######################## HANDLE CURVE GIZMO FINSH ########################	
	var a: MTerrain	
	if active_terrain is MTerrain and event is InputEventMouse:						

		if ray_col.is_collided():			
			col_dis = ray_col.get_collision_position().distance_to(viewport_camera.global_position)
			status_bar.set_height_label(ray_col.get_collision_position().y)
			status_bar.set_distance_label(col_dis)
			status_bar.set_region_label(active_terrain.get_region_id_by_world_pos(ray_col.get_collision_position()))
			if current_edit_mode in [&"sculpt", &"paint"]:				
				if paint_mode_handle(event):
					return true
			if human_male.visible:
				if edit_human_position:
					human_male.global_position = ray_col.get_collision_position()
				if event is InputEventMouseButton:
					if event.button_index == MOUSE_BUTTON_LEFT:
						edit_human_position = false
					if event.button_index == MOUSE_BUTTON_RIGHT and edit_human_position:
						_on_human_male_toggled(false)
					if event.button_index == MOUSE_BUTTON_MIDDLE:
						edit_human_position = true
						
		else:
			col_dis=1000000
			status_bar.disable_height_label()
			status_bar.disable_distance_label()
			status_bar.disable_region_label()
		if col_dis<1000:
			collision_ray_step = (col_dis + 50)/100
		else:
			collision_ray_step = 3
		#set_save_button_disabled(not active_terrain.has_unsave_image())
	if process_input(event):
		return true
	## Fail paint attempt
	## returning the stop so terrain will not be unselected
	#if current_edit_mode == &"paint":		
	#	if event is InputEventMouseButton:
	#		if event.button_mask == MOUSE_BUTTON_LEFT:
	#			return AFTER_GUI_INPUT_STOP



var last_draw_time:int=0
	
func validate_brush():
	if active_object is MTerrain and current_edit_mode == &"paint":		
		if not brush_popup_button.visible: return false
	return true
	
func paint_mode_handle(event:InputEvent):			
	if ray_col.is_collided() and validate_brush():
		brush_decal.visible = true
		brush_decal.set_position(ray_col.get_collision_position())		
		if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				if active_object is MGrass:
					active_object.check_undo()
					undo_redo.create_action("GrassPaint")
					undo_redo.add_undo_method(active_object,"undo")
					undo_redo.commit_action(false)
				elif active_object is MNavigationRegion3D:
					pass
				elif active_object is MTerrain: ## Start of painting						
					active_object.set_brush_start_point(ray_col.get_collision_position(),brush_decal.radius)
					#set_active_layer()
					active_object.images_add_undo_stage()
					undo_redo.create_action("Sculpting")
					undo_redo.add_undo_method(active_object,"images_undo")
					undo_redo.commit_action(false)
			else:
				if mask_decal.is_being_edited:
					mask_decal.is_being_edited = false	
				elif active_object is MGrass:
					active_object.save_grass_data()
				elif active_object is MNavigationRegion3D:
					active_object.save_nav_data()
				elif active_object is MTerrain:
					active_object.save_all_dirty_images()
		if event.button_mask == MOUSE_BUTTON_LEFT:			
			var t = Time.get_ticks_msec()
			var dt = t - last_draw_time			
			last_draw_time = t		
			if mask_decal.is_being_edited:
				return true 
			if draw(ray_col.get_collision_position()):
				return true
		if mask_decal.is_being_edited:			
			if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT:
				mask_decal.is_being_edited = false				
				mask_popup_button.clear_mask()
			mask_decal.set_absolute_terrain_pos(ray_col.get_collision_position())
		elif event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_MIDDLE:
			mask_decal.is_being_edited = true
			return true
	else:
		brush_decal.visible = false
		mask_decal.visible = false
		return false




func on_scene_changed(_root):
	set_edit_mode(null)

func request_hide():		
	set_edit_mode(null, null)
	visible = false
	mpath_gizmo_gui.visible = false
	mcurve_mesh_gui.set_curve_mesh(null)		

func request_show():	
	visible = true
	_on_resized()
	update_edit_mode_options()

func deactivate_editing():	
	if is_instance_valid(edit_mode_button):
		edit_mode_button.text = ""
		edit_mode_button.theme_type_variation = "button_icon_only"
	_on_human_male_toggled(false)
	edit_mode_button.exit_edit_mode_button.visible = false
	brush_decal.visible = false
	mask_decal.visible = false	
	mask_popup_button.clear_mask()	
	var all_terrain =MTerrain.get_all_terrain_nodes()
	for mterrain in all_terrain:
		mterrain.disable_brush_mask()
	paint_panel.visible = false
	mpath_gizmo_gui.visible = false
	mcurve_mesh_gui.visible = false
	brush_popup_button.clear_brushes()
	active_object = null
	current_edit_mode = &""
	
func set_edit_mode(object = active_object, mode=current_edit_mode):	
	if object == active_object and current_edit_mode == mode:	
		return
	
	if object==null or mode ==&"": 
		deactivate_editing()
		return	
	if not is_instance_valid(object):
		push_error("Instance is not valid")
		return
	active_object = object	
	current_edit_mode = mode
	brush_popup_button.visible = true
	mask_popup_button.visible = true
	if object is MPath:		
		#deactivate_editing()
		mpath_gizmo_gui.visible = true
		var active_mterrain = get_active_mterrain()
		if active_mterrain and active_mterrain.is_grid_created():
			mpath_gizmo_gui.set_terrain_snap(active_mterrain)
		else:
			mpath_gizmo_gui.set_terrain_snap(null)
		object.update_gizmos()
	elif object is MCurveMesh:
		mcurve_mesh_gui.set_curve_mesh(object)
		mcurve_mesh_gui.visible = true
	edit_mode_changed.emit(object, mode)	
	var active_mterrain = get_active_mterrain()
	if not active_mterrain: return
	active_mterrain.set_brush_manager(brush_manager)
	mask_popup_button.mterrain = active_mterrain
	mask_popup_button.toggle_grass_settings(false)
	if object is MTerrain:				
		paint_panel.visible = true
		layers_popup_button.visible = true
		grass_merge_sublayer_button.visible = false
		#to do: clean up previous edit mode: grass, nav, and path stuff?, then:		
		for connection in layers_popup_button.get_signal_connection_list("layer_changed"):
			connection.signal.disconnect(connection.callable)					
		if mode == &"sculpt":
			layers_popup_button.init_height_layers(object)
			brush_popup_button.init_height_brushes(brush_manager)
		elif mode == &"paint":
			layers_popup_button.layer_changed.connect(func(id):
				brush_popup_button.visible = id > -1
				mask_popup_button.visible = id > -1
			)
			layers_popup_button.init_color_layers(object, brush_popup_button)
			#Colol layers will init there own brushes
		mask_decal.active_terrain = object
		if not get_active_mterrain().is_grid_created():
			get_active_mterrain().create_grid()
	elif object is MGrass:
		paint_panel.visible = true
		#clean up previous edit mode: grass, nav, and path stuff, then:	
		layers_popup_button.visible = false
		#init_height_layers(object.get_parent())
		brush_popup_button.init_grass_brushes()
		mask_popup_button.toggle_grass_settings(true)
		mask_decal.active_terrain = active_mterrain
		if not object.active or not object.is_init():
			object.active = true
			object.visible = true
			get_active_mterrain().remove_grid()
		if not get_active_mterrain().is_grid_created():
			get_active_mterrain().create_grid()
		grass_merge_sublayer_button.visible = object.has_sublayer()
	elif object is MNavigationRegion3D:
		grass_merge_sublayer_button.visible = false	
		paint_panel.visible = true
		layers_popup_button.visible = false
		brush_popup_button.init_mnavigation_brushes()
		object.set_npoints_visible(true)
		mask_decal.active_terrain = active_mterrain
		if not object.active:
			object.active = true			
			object.visible = true
			get_active_mterrain().remove_grid()
		if not get_active_mterrain().is_grid_created():
			get_active_mterrain().create_grid()
	
	

func draw(brush_position):		
	if active_object is MGrass:
		active_object.draw_grass(brush_position,brush_decal.radius,brush_popup_button.is_grass_add)
		return true
	elif active_object is MNavigationRegion3D:				
		active_object.draw_npoints(brush_position,brush_decal.radius, brush_popup_button.is_grass_add)
		return true
	elif active_object is MTerrain:		
		if current_edit_mode == &"sculpt":										
			active_object.draw_height(brush_position,brush_decal.radius,brush_popup_button.height_brush_id)
			return true
		elif current_edit_mode == &"paint":					
			active_object.draw_color(brush_position,brush_decal.radius,brush_popup_button.color_brush_name,brush_popup_button.color_brush_uniform)			
			return true
		else:
			push_warning("trying to 'draw' on mterrain, but not in sculpt or paint mode")
	else:
		print("draw mterrain fail: active object is ", active_object.name)	


func remove_image(active_terrain, dname):
	var is_grid_created = active_terrain.is_grid_created()
	if is_grid_created:
		active_terrain.remove_grid()	
	var dir = DirAccess.open(active_terrain.dataDir)
	if not dir:
		printerr("Can not open ",active_terrain.dataDir)
		return
	dir.list_dir_begin()
	var file_name :String= dir.get_next()
	var res_names:PackedStringArray = []
	while file_name != "":
		if file_name.get_extension() == "res":
			res_names.append(file_name)
		file_name = dir.get_next()
	#remove config file:
	var config_path = active_terrain.dataDir.path_join(".save_config.ini")
	var config = ConfigFile.new()
	if FileAccess.file_exists(config_path):
		var err = config.load(config_path)
		if err != OK:
			printerr("Can not load config with err ",err)							
	config.erase_section(dname)	
	config.save(config_path)
	
	#remove data
	for res_name in res_names:
		var path = active_terrain.dataDir.path_join(res_name)
		var mres = load(path)
		if not mres is MResource:				
			continue			
		mres.remove_data(dname)
		ResourceSaver.save(mres,path)		
	#remove layers
	var id = 0
	var remove_count = 0
	var layers = active_terrain.brush_layers
	for i in layers.size():
		if layers[i].uniform_name != dname:
			active_terrain.brush_layers[id] = layers[i]
			id += 1		
		else:
			remove_count += 1
	active_terrain.brush_layers_groups_num -= remove_count
			
	
	if is_grid_created:
		active_terrain.create_grid()
	set_edit_mode(null,null)
	#todo: dont' exit edit mode, just change layer selection to 0 or -1



#region responding to signals
func _on_human_male_toggled(button_pressed):	
	edit_human_position = true
	human_male.visible = button_pressed
	if not human_button.button_pressed == button_pressed:
		human_button.button_pressed = button_pressed
	
func _on_info_btn_pressed():
	request_info_window.emit()
	request_info_window

func _on_reload_pressed() -> void:
	get_active_mterrain().create_grid()

func _on_heightmap_import_button_pressed() -> void:
	request_import_window.emit()
	
func _on_image_creator_button_pressed():
	request_image_creator.emit()
#endregion	

#region theme: sizes and colors etc
func _on_resized():			
	if not has_node("VSplitContainer") or not mtools_root:
		call_deferred( "_on_resized" )	
		return
	var vsplit  = $VSplitContainer
	var max_size = get_viewport_rect().size.y / 16 + 2
	var min_size = get_viewport_rect().size.y / 32
	vsplit.size.y = max_size
	vsplit.position.y = -vsplit.size.y
	#print(vsplit.split_offset, " and minsize: ", min_size, " and max: ", max_size)
	vsplit.split_offset = clamp(vsplit.split_offset, -max_size,-min_size)
	resize_children_recursive(mtools_root, clamp(mtools_root.size.y, min_size, max_size ))
	theme.default_font_size = clamp( clamp(mtools_root.size.y, min_size, max_size) /2 , 12,32)
	
func resize_children_recursive(parent, new_size):
	for child in parent.get_children():
		if child is Control and not child is ItemList and not child is LineEdit and not child is Label:
			child.custom_minimum_size.x = new_size
			child.custom_minimum_size.y = new_size		
		resize_children_recursive(child, new_size)

func update_theme():	
	var base_color = EditorInterface.get_editor_settings().get_setting("interface/theme/base_color")	
	var stylebox = preload("res://addons/m_terrain/gui/styles/popup_panel_stylebox.tres")
	if stylebox.bg_color != base_color:
		stylebox.set("bg_color", base_color)	
		
func _on_v_split_container_dragged(offset):
	_on_resized()
#endregion


func _on_grass_merge_sublayer_pressed():
	if not active_object is MGrass:
		push_error("trying to merge grass sublayer, but active object is not MGrass")
		return
	if not active_object.is_init():
		push_error("trying to merge grass sublayer, but grass is not init()")
		return
	if not active_object.has_sublayer():
		push_error("trying to merge grass sublayer, but grass doesn't have sublayer")
		return
	active_object.merge_sublayer() 
	grass_merge_sublayer_button.visible = active_object.has_sublayer()



func _on_walk_terrain_toggled(toggled_on):
	if not get_active_mterrain(): 
		push_error("trying to walk mterrain, but there is no active mterrain")
		walk_terrain_button.button_pressed = false
		return
	walking_terrain = toggled_on
	if toggled_on:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	else:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
