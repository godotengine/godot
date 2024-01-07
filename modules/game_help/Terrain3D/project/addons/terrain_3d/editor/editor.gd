@tool
extends EditorPlugin
#class_name Terrain3DEditorPlugin Cannot be named until Godot #75388


# Includes
const UI: Script = preload("res://addons/terrain_3d/editor/components/ui.gd")
const RegionGizmo: Script = preload("res://addons/terrain_3d/editor/components/region_gizmo.gd")
const TextureDock: Script = preload("res://addons/terrain_3d/editor/components/texture_dock.gd")

var terrain: Terrain3D
var nav_region: NavigationRegion3D

var editor: Terrain3DEditor
var ui: Node # Terrain3DUI see Godot #75388
var texture_dock: TextureDock
var texture_dock_container: CustomControlContainer = CONTAINER_INSPECTOR_BOTTOM

var visible: bool
var region_gizmo: RegionGizmo
var current_region_position: Vector2
var mouse_global_position: Vector3 = Vector3.ZERO


func _enter_tree() -> void:
	editor = Terrain3DEditor.new()
	ui = UI.new()
	ui.plugin = self
	add_child(ui)

	texture_dock = TextureDock.new()
	texture_dock.hide()
	texture_dock.resource_changed.connect(_on_texture_dock_resource_changed)
	texture_dock.resource_inspected.connect(_on_texture_dock_resource_selected)
	texture_dock.resource_selected.connect(ui._on_setting_changed)
	
	region_gizmo = RegionGizmo.new()
	
	add_control_to_container(texture_dock_container, texture_dock)
	texture_dock.get_parent().visibility_changed.connect(_on_texture_dock_visibility_changed)


func _exit_tree() -> void:
	remove_control_from_container(texture_dock_container, texture_dock)
	texture_dock.queue_free()
	ui.queue_free()
	editor.free()

	
func _handles(p_object: Object) -> bool:
	return p_object is Terrain3D or p_object is NavigationRegion3D


func _edit(p_object: Object) -> void:
	if !p_object:
		_clear()
		
	if p_object is Terrain3D:
		if p_object == terrain:
			return
		terrain = p_object
		editor.set_terrain(terrain)
		region_gizmo.set_node_3d(terrain)
		terrain.add_gizmo(region_gizmo)
		terrain.set_plugin(self)
		
		if not terrain.texture_list_changed.is_connected(_load_textures):
			terrain.texture_list_changed.connect(_load_textures)
		_load_textures()
		if not terrain.storage_changed.is_connected(_load_storage):
			terrain.storage_changed.connect(_load_storage)
		_load_storage()
	else:
		terrain = null
	
	if p_object is NavigationRegion3D:
		nav_region = p_object
	else:
		nav_region = null
	
	_update_visibility()

		
func _make_visible(p_visible: bool) -> void:
	visible = p_visible
	_update_visibility()


func _update_visibility() -> void:
	ui.set_visible(visible)
	texture_dock.set_visible(visible and terrain)
	if terrain:
		update_region_grid()
	region_gizmo.set_hidden(not visible or not terrain)


func _clear() -> void:
	if is_terrain_valid():
		terrain.storage_changed.disconnect(_load_storage)
		
		terrain.clear_gizmos()
		terrain = null
		editor.set_terrain(null)
		
	region_gizmo.clear()


func _forward_3d_gui_input(p_viewport_camera: Camera3D, p_event: InputEvent) -> int:
	if not is_terrain_valid():
		return AFTER_GUI_INPUT_PASS
	
	# Handle mouse movement
	if p_event is InputEventMouseMotion:
		if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
			return AFTER_GUI_INPUT_PASS

		## Get mouse location on terrain
		var mouse_pos: Vector2 = p_event.position
		var camera_pos: Vector3 = p_viewport_camera.global_position
		var camera_dir: Vector3 = p_viewport_camera.project_ray_normal(mouse_pos)
		
		# If mouse intersected terrain within 3000 units (3.4e38 is Double max val)
		var intersection_point: Vector3 = terrain.get_intersection(camera_pos, camera_dir)
		if intersection_point.x < 3.4e38:
			mouse_global_position = intersection_point
		else:
			# Else, grab mouse position without considering height
			var t = -Vector3(0, 1, 0).dot(camera_pos) / Vector3(0, 1, 0).dot(camera_dir)
			mouse_global_position = (camera_pos + t * camera_dir)

		# Update decal
		ui.decal.global_position = mouse_global_position
		ui.decal.albedo_mix = 1.0
		ui.decal_timer.start()

		## Update region highlight
		var region_size = terrain.get_storage().get_region_size()
		var region_position: Vector2 = (Vector2(mouse_global_position.x, mouse_global_position.z) / region_size).floor()
		if current_region_position != region_position:
			current_region_position = region_position
			update_region_grid()
			
		if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
			editor.operate(mouse_global_position, p_viewport_camera.rotation.y)
			return AFTER_GUI_INPUT_STOP

	elif p_event is InputEventMouseButton:
		ui.update_decal()
			
		if p_event.get_button_index() == MOUSE_BUTTON_LEFT:
			if p_event.is_pressed():
				if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
					return AFTER_GUI_INPUT_STOP
					
				# If picking
				if ui.picking != Terrain3DEditor.TOOL_MAX: 
					var color: Color
					match ui.picking:
						Terrain3DEditor.HEIGHT:
							color = terrain.get_storage().get_pixel(Terrain3DStorage.TYPE_HEIGHT, mouse_global_position)
						Terrain3DEditor.ROUGHNESS:
							color = terrain.get_storage().get_pixel(Terrain3DStorage.TYPE_COLOR, mouse_global_position)
						Terrain3DEditor.COLOR:
							color = terrain.get_storage().get_color(mouse_global_position)
						_:
							push_error("Unsupported picking type: ", ui.picking)
							return AFTER_GUI_INPUT_STOP
					ui.picking_callback.call(ui.picking, color)
					ui.picking = Terrain3DEditor.TOOL_MAX
					return AFTER_GUI_INPUT_STOP

				# If adjusting regions
				elif editor.get_tool() == Terrain3DEditor.REGION:
					# Skip regions that already exist or don't
					var has_region: bool = terrain.get_storage().has_region(mouse_global_position)
					var op: int = editor.get_operation()
					if	( has_region and op == Terrain3DEditor.ADD) or \
						( not has_region and op == Terrain3DEditor.SUBTRACT ):
						return AFTER_GUI_INPUT_STOP

				# Mouse clicked, start editing
				editor.start_operation(mouse_global_position)
			else:
				# Mouse released, save undo data
				editor.stop_operation()
			return AFTER_GUI_INPUT_STOP
	
	return AFTER_GUI_INPUT_PASS

		
func is_terrain_valid() -> bool:
	var valid: bool = false
	if is_instance_valid(terrain):
		valid = terrain.get_storage() != null
	return valid


func update_texture_dock(p_args: Array) -> void:
	texture_dock.clear()
	
	if is_terrain_valid() and terrain.texture_list:
		var texture_count: int = terrain.texture_list.get_texture_count()
		for i in texture_count:
			var texture: Terrain3DTexture = terrain.texture_list.get_texture(i)
			texture_dock.add_item(texture)
			
		if texture_count < Terrain3DTextureList.MAX_TEXTURES:
			texture_dock.add_item()

	
func update_region_grid() -> void:
	if !region_gizmo.get_node_3d():
		return
		
	if is_terrain_valid():
		region_gizmo.show_rect = editor.get_tool() == Terrain3DEditor.REGION
		region_gizmo.use_secondary_color = editor.get_operation() == Terrain3DEditor.SUBTRACT
		region_gizmo.region_position = current_region_position
		region_gizmo.region_size = terrain.get_storage().get_region_size()
		region_gizmo.grid = terrain.get_storage().get_region_offsets()
		
		terrain.update_gizmos()
		return
		
	region_gizmo.show_rect = false
	region_gizmo.region_size = 1024
	region_gizmo.grid = [Vector2i.ZERO]


# Signal handlers


func _load_textures() -> void:
	if terrain and terrain.texture_list:
		if not terrain.texture_list.textures_changed.is_connected(update_texture_dock):
			terrain.texture_list.textures_changed.connect(update_texture_dock)
		update_texture_dock(Array())				


func _load_storage() -> void:
	if terrain:
		update_region_grid()


func _on_texture_dock_resource_changed(texture: Resource, index: int) -> void:
	if is_terrain_valid():
		# If removing last entry and its selected, clear inspector
		if not texture and index == texture_dock.get_selected_index() and \
				texture_dock.get_selected_index() == texture_dock.entries.size() - 2:
			get_editor_interface().inspect_object(null)			
		terrain.get_texture_list().set_texture(index, texture)
		call_deferred("_load_storage")


func _on_texture_dock_resource_selected(texture) -> void:
	get_editor_interface().inspect_object(texture, "", true)


func _on_texture_dock_visibility_changed() -> void:
	if texture_dock.get_parent() != null:
		remove_control_from_container(texture_dock_container, texture_dock)
	
	if texture_dock.get_parent() == null:
		texture_dock_container = CONTAINER_INSPECTOR_BOTTOM
		if get_editor_interface().is_distraction_free_mode_enabled():
			texture_dock_container = CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT
		add_control_to_container(texture_dock_container, texture_dock)
