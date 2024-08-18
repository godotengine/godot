@tool
extends EditorPlugin
var version:String="0.14.1 alpha"

var tools= null

var current_main_screen_name =""

var tsnap = null

var raw_img_importer = null
var raw_tex_importer = null

var active_snap_object:Node3D = null
var last_camera_position:Vector3

var collision_ray_step=0.2
var ray_col:MCollision
var col_dis:float
var is_paint_active:bool = false

var action=""

var current_window_info=null

var gizmo_moctmesh
var gizmo_mpath
var gizmo_mpath_gui
var mcurve_mesh_gui

var inspector_mpath

#region keyboard actions
var default_keyboard_actions 

const setting_path = 'addons/m_terrain/keymap/'

func add_keymap(force_default = false):	
	set_default_keymap()
	for action in default_keyboard_actions:
		var path = setting_path + action.name
		if force_default or not ProjectSettings.has_setting(path):
			var a = InputEventKey.new()
			a.keycode = action.keycode
			a.pressed = action.pressed
			if "shift" in action.keys():
				a.shift_pressed = action.shift				
			ProjectSettings.set_setting(path, a)
		var e = ProjectSettings.get_setting(path)
		if not InputMap.has_action(action.name):			
			InputMap.add_action(action.name)		
		for i in default_keyboard_actions.size():
			if default_keyboard_actions[i].name == action.name:
				default_keyboard_actions[i].keycode = e.keycode
				default_keyboard_actions[i].shift = e.shift_pressed
				default_keyboard_actions[i].ctrl = e.ctrl_pressed
				default_keyboard_actions[i].alt = e.alt_pressed
		InputMap.action_add_event(action.name, e)
	
func remove_keymap():
	for action in default_keyboard_actions:
		InputMap.erase_action(action.name)
#endregion

#region init and de-init
func _enter_tree():		
	if Engine.is_editor_hint():		
		var main_screen = EditorInterface.get_editor_main_screen()											
		main_screen_changed.connect(_on_main_screen_changed)		
	
		#add_tool_menu_item("MTerrain import/export", show_import_window)
		#add_tool_menu_item("MTerrain image create/remove", show_image_creator_window)
		
		tools = preload("res://addons/m_terrain/gui/mtools.tscn").instantiate()		
		tools.request_info_window.connect(show_info_window)
		tools.request_import_window.connect(show_import_window)
		tools.request_image_creator.connect(show_image_creator_window)
		tools.edit_mode_changed.connect(select_object)		
		main_screen.add_child(tools)

		get_tree().node_added.connect(tools.on_node_modified)
		get_tree().node_renamed.connect(tools.on_node_modified)
		get_tree().node_removed.connect(tools.on_node_modified)
		
		tools.set_brush_decal( preload("res://addons/m_terrain/gui/brush_decal.tscn").instantiate()	)
		main_screen.add_child(tools.brush_decal)
		
		tools.set_mask_decal( preload("res://addons/m_terrain/gui/mask_decal.tscn").instantiate() )
		main_screen.add_child(tools.mask_decal)
		
		tools.human_male = preload("res://addons/m_terrain/gui/human_male.tscn").instantiate()
		main_screen.add_child(tools.human_male)
		tools.human_male.visible = false
		
		MTool.enable_editor_plugin()
		
		get_editor_interface().get_selection().selection_changed.connect(selection_changed)
		
		tsnap = preload("res://addons/m_terrain/gui/tsnap.tscn").instantiate()
		tsnap.pressed.connect(func(): tsnap_pressed(tools.get_active_mterrain()))
		tsnap.visible = false
		add_control_to_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_MENU,tsnap)				
				
		###### GIZMO
		gizmo_moctmesh = preload("res://addons/m_terrain/gizmos/moct_mesh_gizmo.gd").new()
		gizmo_mpath = preload("res://addons/m_terrain/gizmos/mpath_gizmo.gd").new()
		gizmo_mpath_gui = tools.find_child("mpath_gizmo_gui") #load("res://addons/m_terrain/gizmos/mpath_gizmo_gui.tscn").instantiate()
		mcurve_mesh_gui = tools.find_child("mcurve_mesh") #load("res://addons/m_terrain/gizmos/mcurve_mesh_gui.tscn").instantiate()
		add_node_3d_gizmo_plugin(gizmo_moctmesh)
		gizmo_mpath.ur = get_undo_redo()
		add_node_3d_gizmo_plugin(gizmo_mpath)		
		gizmo_mpath.set_gui(gizmo_mpath_gui)
		gizmo_mpath.mterrain_plugin = self		
		#### Inspector
		inspector_mpath = preload("res://addons/m_terrain/inspector/mpath.gd").new()
		inspector_mpath.gizmo = gizmo_mpath
		add_inspector_plugin(inspector_mpath)
				
		add_keymap()		
		
func _ready() -> void:	
	EditorInterface.set_main_screen_editor("Script")
	EditorInterface.set_main_screen_editor("3D")
	
func _exit_tree():	
	if Engine.is_editor_hint():
		remove_keymap()	
		remove_tool_menu_item("MTerrain import/export")
		remove_tool_menu_item("MTerrain image create/remove")		
		tools.brush_decal.queue_free()
		tools.mask_decal.queue_free()		
		tsnap.queue_free()
		tools.human_male.queue_free()
		
		get_tree().node_added.disconnect(tools.on_node_modified)
		get_tree().node_renamed.disconnect(tools.on_node_modified)
		get_tree().node_removed.disconnect(tools.on_node_modified)
		tools.queue_free()
		
		###### GIZMO
		remove_node_3d_gizmo_plugin(gizmo_moctmesh)
		remove_node_3d_gizmo_plugin(gizmo_mpath)
		#remove_control_from_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_MENU,gizmo_mpath_gui)
		#remove_control_from_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_MENU,mcurve_mesh_gui)
		### Inspector
		remove_inspector_plugin(inspector_mpath)
#endregion

func _on_main_screen_changed(screen_name):
	current_main_screen_name = screen_name	
	selection_changed()	

#select_object is called when tools changes edit mode
func select_object(object, mode):	
	EditorInterface.get_selection().clear()
	EditorInterface.get_selection().add_node(object)

func selection_changed():
	var selection = get_editor_interface().get_selection().get_selected_nodes()

	#TO DO: decide if this behaviour is good.
	if selection.size() != 1:				
		tools.request_hide()
		gizmo_mpath_gui.visible = false
		mcurve_mesh_gui.set_curve_mesh(null)		
		return
		
	if not tools or not is_instance_valid(EditorInterface.get_edited_scene_root()): return
	if not current_main_screen_name == "3D":
		tools.request_hide()
		return	
	if selection[0] is MTerrain:
		tools.request_show()	
		return
	if selection[0].get_parent() is MTerrain:
		if selection[0] is MGrass or selection[0] is MNavigationRegion3D:
			tools.request_show()	
			return
	if selection[0] is MPath or selection[0] is MCurveMesh:
		tools.request_show()	
		return
	if mcurve_mesh_gui.obj and mcurve_mesh_gui.is_active():
		tools.request_show()	
		return
	tools.request_hide()

func _handles(object):
	if not Engine.is_editor_hint(): return false
	if not current_main_screen_name == "3D":
		tools.request_hide()
		return false
	if mcurve_mesh_gui and mcurve_mesh_gui.obj and mcurve_mesh_gui.is_active(): return false
	
	active_snap_object = null
	tsnap.visible = false
	
	if object is MPath:
		tools.set_active_object(object)
		tools.request_show()		
		#gizmo_mpath_gui.visible = true		
		return true
	elif gizmo_mpath_gui:
		gizmo_mpath_gui.visible = false
	
	if object is MCurveMesh:
		#mcurve_mesh_gui.set_curve_mesh(object)
		tools.set_active_object(object)
		tools.request_show()	
		return true
	else:
		mcurve_mesh_gui.set_curve_mesh(null)
	
	if object is MTerrain:		
		tools.set_active_object(object)			
		tools.request_show()		
		return true
	
	if object is MGrass and object.get_parent() is MTerrain:			
		tools.set_active_object(object)
		tools.request_show()			
		return true
	
	if object is MNavigationRegion3D and object.get_parent() is MTerrain:
		tools.set_active_object(object)
		tools.request_show()
		return true
	
	if object is MCurveMesh and object.get_parent() is MTerrain:
		tools.set_active_object(object)
		tools.request_show()
		return true
	else:
		#for some reason these get selected when switching from grass paint mode to terrain sculpt/paint mode:
		if object is MTerrainMaterial or object is MBrushLayers:
			return false
		tools.request_hide()		
		#TO DO: fix snap tool setting of active terain		
		if object is Node3D:
			for mterrain:MTerrain in tools.get_all_mterrain(EditorInterface.get_edited_scene_root()):
				if mterrain.is_grid_created():
					active_snap_object = object
					tsnap.visible = true
		return false

func _forward_3d_gui_input(viewport_camera, event):
	if not is_instance_valid(EditorInterface.get_edited_scene_root()): 
		return AFTER_GUI_INPUT_PASS
		
	var active_terrain = tools.get_active_mterrain()
	if not active_terrain is MTerrain: 
		ray_col = null
	elif event is InputEventMouse:
		var ray:Vector3 = viewport_camera.project_ray_normal(event.position)
		var pos:Vector3 = viewport_camera.global_position
		ray_col = active_terrain.get_ray_collision_point(pos,ray,collision_ray_step,1000)
	
	if tools.walking_terrain:
		tools.editor_camera = viewport_camera
		if tools.process_input_terrain_walk(viewport_camera, event):
			return AFTER_GUI_INPUT_STOP
	
	
	for terrain in tools.get_all_mterrain():
		terrain.set_editor_camera(viewport_camera)	
	######################## HANDLE CURVE GIZMO ##############################
	if gizmo_mpath_gui.visible:				
		return gizmo_mpath._forward_3d_gui_input(viewport_camera, event, ray_col)
	######################## HANDLE CURVE GIZMO FINSH ########################	
	
	if active_terrain is MTerrain and event is InputEventMouse:						
		if ray_col.is_collided():			
			col_dis = ray_col.get_collision_position().distance_to(viewport_camera.global_position)
			tools.status_bar.set_height_label(ray_col.get_collision_position().y)
			tools.status_bar.set_distance_label(col_dis)
			tools.status_bar.set_region_label(active_terrain.get_region_id_by_world_pos(ray_col.get_collision_position()))
			if tools.current_edit_mode in [&"sculpt", &"paint"]:							
				if paint_mode_handle(event):
					return AFTER_GUI_INPUT_STOP			
			if tools.human_male.visible:
				if tools.edit_human_position:
					tools.human_male.global_position = ray_col.get_collision_position()
				if event is InputEventMouseButton:
					if event.button_index == MOUSE_BUTTON_LEFT:
						tools.edit_human_position = false
					if event.button_index == MOUSE_BUTTON_RIGHT and tools.edit_human_position:
						tools._on_human_male_toggled(false)
					if event.button_index == MOUSE_BUTTON_MIDDLE:
						tools.edit_human_position = true
						
		else:
			col_dis=1000000
			tools.status_bar.disable_height_label()
			tools.status_bar.disable_distance_label()
			tools.status_bar.disable_region_label()
		if col_dis<1000:
			collision_ray_step = (col_dis + 50)/100
		else:
			collision_ray_step = 3
		#tools.set_save_button_disabled(not active_terrain.has_unsave_image())
	if tools.process_input(event):
		return AFTER_GUI_INPUT_STOP
	## Fail paint attempt
	## returning the stop so terrain will not be unselected
	#if tools.current_edit_mode == &"paint":		
	#	if event is InputEventMouseButton:
	#		if event.button_mask == MOUSE_BUTTON_LEFT:
	#			return AFTER_GUI_INPUT_STOP



var last_draw_time:int=0
	
func paint_mode_handle(event:InputEvent):	
	if ray_col.is_collided():
		tools.brush_decal.visible = true
		tools.brush_decal.set_position(ray_col.get_collision_position())		
		if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				if tools.active_object is MGrass:
					tools.active_object.check_undo()
					get_undo_redo().create_action("GrassPaint")
					get_undo_redo().add_undo_method(tools.active_object,"undo")
					get_undo_redo().commit_action(false)
				elif tools.active_object is MNavigationRegion3D:
					pass
				elif tools.active_object is MTerrain: ## Start of painting						
					tools.active_object.set_brush_start_point(ray_col.get_collision_position(),tools.brush_decal.radius)
					#tools.set_active_layer()
					tools.active_object.images_add_undo_stage()
					get_undo_redo().create_action("Sculpting")
					get_undo_redo().add_undo_method(tools.active_object,"images_undo")
					get_undo_redo().commit_action(false)
			else:
				if tools.mask_decal.is_being_edited:
					tools.mask_decal.is_being_edited = false	
				elif tools.active_object is MGrass:
					tools.active_object.save_grass_data()
				elif tools.active_object is MNavigationRegion3D:
					tools.active_object.save_nav_data()
				elif tools.active_object is MTerrain:
					tools.active_object.save_all_dirty_images()
		if event.button_mask == MOUSE_BUTTON_LEFT:			
			var t = Time.get_ticks_msec()
			var dt = t - last_draw_time			
			last_draw_time = t		
			if tools.mask_decal.is_being_edited:
				return AFTER_GUI_INPUT_STOP 
			if tools.draw(ray_col.get_collision_position()):
				return AFTER_GUI_INPUT_STOP 
		if tools.mask_decal.is_being_edited:			
			if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT:
				tools.mask_decal.is_being_edited = false				
				tools.mask_popup_button.clear_mask()
			tools.mask_decal.set_absolute_terrain_pos(ray_col.get_collision_position())
		elif event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_MIDDLE:
			tools.mask_decal.is_being_edited = true
			AFTER_GUI_INPUT_STOP
	else:
		tools.brush_decal.visible = false
		tools.mask_decal.visible = false
		return AFTER_GUI_INPUT_PASS



#To do: fix tsnap pressed - how does it find active_terrain?
func tsnap_pressed(active_terrain:MTerrain):
	if active_terrain and active_snap_object and active_terrain.is_grid_created():
		var h:float = active_terrain.get_height(active_snap_object.global_position)
		active_snap_object.global_position.y = h

func show_import_window():	
	var window = preload("res://addons/m_terrain/gui/import_window.tscn").instantiate()
	add_child(window)
	if tools.get_active_mterrain() is MTerrain:
		window.init_export(tools.get_active_mterrain())

func show_image_creator_window():	
	if tools.get_active_mterrain() is MTerrain:	
		var window = preload("res://addons/m_terrain/gui/image_creator_window.tscn").instantiate()
		add_child(window)
		window.set_terrain(tools.get_active_mterrain())

func show_info_window(active_terrain:MTerrain = tools.get_active_mterrain()):
	if is_instance_valid(current_window_info):
		current_window_info.queue_free()
	current_window_info = preload("res://addons/m_terrain/gui/terrain_info.tscn").instantiate()
	add_child(current_window_info)
	current_window_info.generate_info(active_terrain,version, default_keyboard_actions)
	current_window_info.keymap_changed.connect(update_keymap)
	current_window_info.restore_default_keymap_requested.connect(func():
		add_keymap(true)
		current_window_info.create_keymapping_interface(default_keyboard_actions)
	)

func set_default_keymap():
	default_keyboard_actions = [
		{"name": "mterrain_brush_size_increase", "keycode": KEY_BRACKETRIGHT, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_brush_size_decrease", "keycode": KEY_BRACKETLEFT, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		
		{"name": "mterrain_mask_size_increase", "keycode": KEY_PERIOD, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_mask_size_decrease", "keycode": KEY_COMMA, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_mask_rotate_clockwise", "keycode": KEY_L, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_mask_rotate_counter_clockwise", "keycode": KEY_K, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_mask_rotation_reset", "keycode": KEY_SEMICOLON, "pressed": true, "shift": false, "ctrl": false, "alt": false},

		{"name": "mterrain_walk_forward", "keycode": KEY_W, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_walk_backward", "keycode": KEY_S, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_walk_left", "keycode": KEY_A, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mterrain_walk_right", "keycode": KEY_D, "pressed": true, "shift": false, "ctrl": false, "alt": false},


		{"name": "mpath_toggle_mode", "keycode": KEY_QUOTELEFT, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_toggle_mirror", "keycode": KEY_M, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_toggle_mirror_length", "keycode": KEY_L, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		
		{"name": "mpath_validate", "keycode": KEY_P, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_select_linked", "keycode": KEY_L, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_swap_points", "keycode": KEY_T, "pressed": true, "shift": true, "ctrl": false, "alt": false},
		{"name": "mpath_toggle_connection", "keycode": KEY_T, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_remove_point", "keycode": KEY_BACKSPACE, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_disconnect_point", "keycode": KEY_B, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_connect_point", "keycode": KEY_C, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_tilt_mode", "keycode": KEY_R, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_scale_mode", "keycode": KEY_K, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_lock_zy", "keycode": KEY_X, "pressed": true, "shift": true, "ctrl": false, "alt": false},
		{"name": "mpath_lock_xz", "keycode": KEY_Y, "pressed": true, "shift": true, "ctrl": false, "alt": false},
		{"name": "mpath_lock_xy", "keycode": KEY_Z, "pressed": true, "shift": true, "ctrl": false, "alt": false},
		{"name": "mpath_lock_x", "keycode": KEY_X, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_lock_y", "keycode": KEY_Y, "pressed": true, "shift": false, "ctrl": false, "alt": false},
		{"name": "mpath_lock_z", "keycode": KEY_Z, "pressed": true,  "shift": false, "ctrl": false, "alt": false },
	]
func update_keymap(who, keycode, ctrl, alt, shift):
	var a = InputEventKey.new()
	a.keycode = keycode
	a.pressed = true	
	a.ctrl_pressed = ctrl
	a.alt_pressed = alt
	a.shift_pressed = shift	
	ProjectSettings.set_setting(setting_path + who, a)
	InputMap.action_erase_events(who)
	InputMap.action_add_event(who, a)
	for i in default_keyboard_actions.size():
		if default_keyboard_actions[i].name == who:						
			default_keyboard_actions[i].keycode = keycode
			default_keyboard_actions[i].ctrl = ctrl
			default_keyboard_actions[i].alt = alt
			default_keyboard_actions[i].shift = shift			
