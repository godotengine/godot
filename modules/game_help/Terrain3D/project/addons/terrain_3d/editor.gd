@tool
extends EditorPlugin
#class_name Terrain3DEditorPlugin Cannot be named until Godot #75388


# Includes
const UI: Script = preload("res://addons/terrain_3d/src/ui.gd")
const RegionGizmo: Script = preload("res://addons/terrain_3d/src/region_gizmo.gd")
const ASSET_DOCK: String = "res://addons/terrain_3d/src/asset_dock.tscn"
const PS_DOCK_POSITION: String = "terrain3d/config/dock_position"
const PS_DOCK_PINNED: String = "terrain3d/config/dock_pinned"

var terrain: Terrain3D
var _last_terrain: Terrain3D
var nav_region: NavigationRegion3D

var editor: Terrain3DEditor
var ui: Node # Terrain3DUI see Godot #75388
var asset_dock: PanelContainer
var region_gizmo: RegionGizmo
var visible: bool
var current_region_position: Vector2
var mouse_global_position: Vector3 = Vector3.ZERO

# Track negative input (CTRL)
var _negative_input: bool = false
# Track state prior to pressing CTRL: -1 not tracked, 0 false, 1 true
var _prev_enable_state: int = -1


func _enter_tree() -> void:
	editor = Terrain3DEditor.new()
	ui = UI.new()
	ui.plugin = self
	add_child(ui)

	region_gizmo = RegionGizmo.new()

	scene_changed.connect(_on_scene_changed)

	asset_dock = load(ASSET_DOCK).instantiate()
	asset_dock.initialize(self)

	
func _exit_tree() -> void:
	asset_dock.remove_dock(true)
	asset_dock.queue_free()
	ui.queue_free()
	editor.free()

	scene_changed.disconnect(_on_scene_changed)


func _handles(p_object: Object) -> bool:
	if p_object is Terrain3D:
		return true
	if is_instance_valid(_last_terrain) and _last_terrain.is_inside_tree():
		if p_object is NavigationRegion3D:
			return true
		if p_object is Terrain3DObjects or (p_object is Node3D and p_object.get_parent() is Terrain3DObjects):
			return true
	return false


func _edit(p_object: Object) -> void:
	if !p_object:
		_clear()

	if p_object is Terrain3D:
		if p_object == terrain:
			return
		terrain = p_object
		_last_terrain = terrain
		editor.set_terrain(terrain)
		region_gizmo.set_node_3d(terrain)
		terrain.add_gizmo(region_gizmo)
		terrain.set_plugin(self)
		
		# Connect to new Assets resource
		if not terrain.assets_changed.is_connected(asset_dock.update_assets):
			terrain.assets_changed.connect(asset_dock.update_assets)
		asset_dock.update_assets()
		# Connect to new Storage resource
		if not terrain.storage_changed.is_connected(_load_storage):
			terrain.storage_changed.connect(_load_storage)
		_load_storage()
	else:
		terrain = null

	if is_instance_valid(_last_terrain) and _last_terrain.is_inside_tree():
		if p_object is NavigationRegion3D:
			nav_region = p_object
		else:
			nav_region = null

		if p_object is Terrain3DObjects:
			p_object.editor_setup(self)
		elif p_object is Node3D and p_object.get_parent() is Terrain3DObjects:
			p_object.get_parent().editor_setup(self)
	
		
func _make_visible(p_visible: bool, p_redraw: bool = false) -> void:
	visible = p_visible
	ui.set_visible(visible)
	update_region_grid()
	asset_dock.update_dock(visible)


func _clear() -> void:
	if is_terrain_valid():
		terrain.storage_changed.disconnect(_load_storage)
		
		terrain.clear_gizmos()
		terrain = null
		editor.set_terrain(null)
		
		ui.clear_picking()
		
	region_gizmo.clear()


func _forward_3d_gui_input(p_viewport_camera: Camera3D, p_event: InputEvent) -> int:
	if not is_terrain_valid():
		return AFTER_GUI_INPUT_PASS
	
	## Track negative input (CTRL)
	if p_event is InputEventKey and not p_event.echo and p_event.keycode == KEY_CTRL:
		if p_event.is_pressed():
			_negative_input = true
			_prev_enable_state = int(ui.toolbar_settings.get_setting("enable"))
			ui.toolbar_settings.set_setting("enable", false)
		else:
			_negative_input = false
			ui.toolbar_settings.set_setting("enable", bool(_prev_enable_state))
			_prev_enable_state = -1
	
	## Handle mouse movement
	if p_event is InputEventMouseMotion:
		if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
			return AFTER_GUI_INPUT_PASS

		if _prev_enable_state >= 0 and not Input.is_key_pressed(KEY_CTRL):
			_negative_input = false
			ui.toolbar_settings.set_setting("enable", bool(_prev_enable_state))
			_prev_enable_state = -1

		## Setup for active camera & viewport
		
		# Snap terrain to current camera 
		terrain.set_camera(p_viewport_camera)

		# Detect if viewport is set to half_resolution
		# Structure is: Node3DEditorViewportContainer/Node3DEditorViewport(4)/SubViewportContainer/SubViewport/Camera3D
		var editor_vpc: SubViewportContainer = p_viewport_camera.get_parent().get_parent()
		var full_resolution: bool = false if editor_vpc.stretch_shrink == 2 else true

		## Get mouse location on terrain
		
		# Project 2D mouse position to 3D position and direction
		var mouse_pos: Vector2 = p_event.position if full_resolution else p_event.position/2
		var camera_pos: Vector3 = p_viewport_camera.project_ray_origin(mouse_pos)
		var camera_dir: Vector3 = p_viewport_camera.project_ray_normal(mouse_pos)

		# If region tool, grab mouse position without considering height
		if editor.get_tool() == Terrain3DEditor.REGION:
			var t = -Vector3(0, 1, 0).dot(camera_pos) / Vector3(0, 1, 0).dot(camera_dir)
			mouse_global_position = (camera_pos + t * camera_dir)
		else:			
			# Else look for intersection with terrain
			var intersection_point: Vector3 = terrain.get_intersection(camera_pos, camera_dir)
			if intersection_point.z > 3.4e38: # double max
				return AFTER_GUI_INPUT_STOP
			mouse_global_position = intersection_point
		
		## Update decal
		ui.decal.global_position = mouse_global_position
		ui.decal.albedo_mix = 1.0
		if ui.decal_timer.is_stopped():
			ui.update_decal()
		else:
			ui.decal_timer.start()

		## Update region highlight
		var region_size = terrain.get_storage().get_region_size()
		var region_position: Vector2 = ( Vector2(mouse_global_position.x, mouse_global_position.z) \
			/ (region_size * terrain.get_mesh_vertex_spacing()) ).floor()
		if current_region_position != region_position:
			current_region_position = region_position
			update_region_grid()
			
		if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT) and editor.is_operating():
			editor.operate(mouse_global_position, p_viewport_camera.rotation.y)
			return AFTER_GUI_INPUT_STOP

	elif p_event is InputEventMouseButton:
		ui.update_decal()
			
		if p_event.get_button_index() == MOUSE_BUTTON_LEFT:
			if p_event.is_pressed():
				if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
					return AFTER_GUI_INPUT_STOP
					
				# If picking
				if ui.is_picking():
					ui.pick(mouse_global_position)
					if not ui.operation_builder or not ui.operation_builder.is_ready():
						return AFTER_GUI_INPUT_STOP
				
				# If adjusting regions
				if editor.get_tool() == Terrain3DEditor.REGION:
					# Skip regions that already exist or don't
					var has_region: bool = terrain.get_storage().has_region(mouse_global_position)
					var op: int = editor.get_operation()
					if	( has_region and op == Terrain3DEditor.ADD) or \
						( not has_region and op == Terrain3DEditor.SUBTRACT ):
						return AFTER_GUI_INPUT_STOP
				
				# If an automatic operation is ready to go (e.g. gradient)
				if ui.operation_builder and ui.operation_builder.is_ready():
					ui.operation_builder.apply_operation(editor, mouse_global_position, p_viewport_camera.rotation.y)
					return AFTER_GUI_INPUT_STOP
				
				# Mouse clicked, start editing
				editor.start_operation(mouse_global_position)
				editor.operate(mouse_global_position, p_viewport_camera.rotation.y)
				return AFTER_GUI_INPUT_STOP
			
			elif editor.is_operating():
				# Mouse released, save undo data
				editor.stop_operation()
				return AFTER_GUI_INPUT_STOP
	
	return AFTER_GUI_INPUT_PASS

	
func _load_storage() -> void:
	if terrain:
		update_region_grid()


func update_region_grid() -> void:
	if not region_gizmo:
		return

	region_gizmo.set_hidden(not visible)
	
	if is_terrain_valid():
		region_gizmo.show_rect = editor.get_tool() == Terrain3DEditor.REGION
		region_gizmo.use_secondary_color = editor.get_operation() == Terrain3DEditor.SUBTRACT
		region_gizmo.region_position = current_region_position
		region_gizmo.region_size = terrain.get_storage().get_region_size() * terrain.get_mesh_vertex_spacing()
		region_gizmo.grid = terrain.get_storage().get_region_offsets()
		
		terrain.update_gizmos()
		return
		
	region_gizmo.show_rect = false
	region_gizmo.region_size = 1024
	region_gizmo.grid = [Vector2i.ZERO]


func _on_scene_changed(scene_root: Node) -> void:
	if not scene_root:
		return
		
	for node in scene_root.find_children("", "Terrain3DObjects"):
		node.editor_setup(self)

	asset_dock.update_assets()
	await get_tree().create_timer(2).timeout
	asset_dock.update_thumbnails()

		
func is_terrain_valid(p_terrain: Terrain3D = null) -> bool:
	var t: Terrain3D
	if p_terrain:
		t = p_terrain
	else:
		t = terrain
	if is_instance_valid(t) and t.is_inside_tree() and t.get_storage():
		return true
	return false


func is_selected() -> bool:
	var selected: Array[Node] = get_editor_interface().get_selection().get_selected_nodes()
	for node in selected:
		if node.get_instance_id() == _last_terrain.get_instance_id():
			return true
			
	return false	


func select_terrain() -> void:
	if is_instance_valid(_last_terrain) and is_terrain_valid(_last_terrain) and not is_selected():
		var es: EditorSelection = get_editor_interface().get_selection()
		es.clear()
		es.add_node(_last_terrain)
