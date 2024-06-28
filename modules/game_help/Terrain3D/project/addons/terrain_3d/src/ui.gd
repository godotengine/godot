extends Node
#class_name Terrain3DUI Cannot be named until Godot #75388


# Includes
const Toolbar: Script = preload("res://addons/terrain_3d/src/toolbar.gd")
const ToolSettings: Script = preload("res://addons/terrain_3d/src/tool_settings.gd")
const TerrainTools: Script = preload("res://addons/terrain_3d/src/terrain_tools.gd")
const OperationBuilder: Script = preload("res://addons/terrain_3d/src/operation_builder.gd")
const GradientOperationBuilder: Script = preload("res://addons/terrain_3d/src/gradient_operation_builder.gd")
const COLOR_RAISE := Color.WHITE
const COLOR_LOWER := Color.BLACK
const COLOR_SMOOTH := Color(0.5, 0, .1)
const COLOR_EXPAND := Color.ORANGE
const COLOR_REDUCE := Color.BLUE_VIOLET
const COLOR_FLATTEN := Color(0., 0.32, .4)
const COLOR_SLOPE := Color.YELLOW
const COLOR_PAINT := Color.FOREST_GREEN
const COLOR_SPRAY := Color.SEA_GREEN
const COLOR_ROUGHNESS := Color.ROYAL_BLUE
const COLOR_AUTOSHADER := Color.DODGER_BLUE
const COLOR_HOLES := Color.BLACK
const COLOR_NAVIGATION := Color.REBECCA_PURPLE
const COLOR_INSTANCER := Color.CRIMSON
const COLOR_PICK_COLOR := Color.WHITE
const COLOR_PICK_HEIGHT := Color.DARK_RED
const COLOR_PICK_ROUGH := Color.ROYAL_BLUE

const RING1: String = "res://addons/terrain_3d/brushes/ring1.exr"
@onready var ring_texture := ImageTexture.create_from_image(Terrain3DUtil.black_to_alpha(Image.load_from_file(RING1)))

var plugin: EditorPlugin # Actually Terrain3DEditorPlugin, but Godot still has CRC errors
var toolbar: Toolbar
var toolbar_settings: ToolSettings
var terrain_tools: TerrainTools
var setting_has_changed: bool = false
var visible: bool = false
var picking: int = Terrain3DEditor.TOOL_MAX
var picking_callback: Callable
var decal: Decal
var decal_timer: Timer
var gradient_decals: Array[Decal]
var brush_data: Dictionary
var operation_builder: OperationBuilder


func _enter_tree() -> void:
	toolbar = Toolbar.new()
	toolbar.hide()
	toolbar.connect("tool_changed", _on_tool_changed)
	
	toolbar_settings = ToolSettings.new()
	toolbar_settings.connect("setting_changed", _on_setting_changed)
	toolbar_settings.connect("picking", _on_picking)
	toolbar_settings.hide()

	terrain_tools = TerrainTools.new()
	terrain_tools.plugin = plugin
	terrain_tools.hide()

	plugin.add_control_to_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_SIDE_LEFT, toolbar)
	plugin.add_control_to_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_BOTTOM, toolbar_settings)
	plugin.add_control_to_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_MENU, terrain_tools)

	_on_tool_changed(Terrain3DEditor.REGION, Terrain3DEditor.ADD)
	
	decal = Decal.new()
	add_child(decal)
	decal_timer = Timer.new()
	decal_timer.wait_time = .5
	decal_timer.one_shot = true
	decal_timer.timeout.connect(Callable(func(node):
		if node:
			get_tree().create_tween().tween_property(node, "albedo_mix", 0.0, 0.15)).bind(decal))
	add_child(decal_timer)


func _exit_tree() -> void:
	plugin.remove_control_from_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_SIDE_LEFT, toolbar)
	plugin.remove_control_from_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_BOTTOM, toolbar_settings)
	toolbar.queue_free()
	toolbar_settings.queue_free()
	terrain_tools.queue_free()
	decal.queue_free()
	decal_timer.queue_free()
	for gradient_decal in gradient_decals:
		gradient_decal.queue_free()
	gradient_decals.clear()


func set_visible(p_visible: bool) -> void:
	visible = p_visible
	terrain_tools.set_visible(p_visible)
	toolbar.set_visible(p_visible)
	toolbar_settings.set_visible(p_visible)
	update_decal()


func set_menu_visibility(p_list: Control, p_visible: bool) -> void:
	if p_list:
		p_list.get_parent().get_parent().visible = p_visible
	

func _on_tool_changed(p_tool: Terrain3DEditor.Tool, p_operation: Terrain3DEditor.Operation) -> void:
	clear_picking()
	set_menu_visibility(toolbar_settings.advanced_list, true)
	set_menu_visibility(toolbar_settings.scale_list, false)
	set_menu_visibility(toolbar_settings.rotation_list, false)
	set_menu_visibility(toolbar_settings.height_list, false)
	set_menu_visibility(toolbar_settings.color_list, false)

	# Select which settings to show. Options in tool_settings.gd:_ready
	var to_show: PackedStringArray = []
	
	match p_tool:
		Terrain3DEditor.HEIGHT:
			to_show.push_back("brush")
			to_show.push_back("size")
			to_show.push_back("strength")
			if p_operation == Terrain3DEditor.REPLACE:
				to_show.push_back("height")
				to_show.push_back("height_picker")
			if p_operation == Terrain3DEditor.GRADIENT:
				to_show.push_back("gradient_points")
				to_show.push_back("drawable")
		
		Terrain3DEditor.TEXTURE:
			to_show.push_back("brush")
			to_show.push_back("size")
			to_show.push_back("enable_texture")
			if p_operation == Terrain3DEditor.ADD:
				to_show.push_back("strength")
			to_show.push_back("enable_angle")
			to_show.push_back("angle")
			to_show.push_back("angle_picker")
			to_show.push_back("dynamic_angle")
			to_show.push_back("enable_scale")
			to_show.push_back("scale")
			to_show.push_back("scale_picker")

		Terrain3DEditor.COLOR:
			to_show.push_back("brush")
			to_show.push_back("size")
			to_show.push_back("strength")
			to_show.push_back("color")
			to_show.push_back("color_picker")

		Terrain3DEditor.ROUGHNESS:
			to_show.push_back("brush")
			to_show.push_back("size")
			to_show.push_back("strength")
			to_show.push_back("roughness")
			to_show.push_back("roughness_picker")

		Terrain3DEditor.AUTOSHADER, Terrain3DEditor.HOLES, Terrain3DEditor.NAVIGATION:
			to_show.push_back("brush")
			to_show.push_back("size")
			to_show.push_back("enable")

		Terrain3DEditor.INSTANCER:
			to_show.push_back("size")
			to_show.push_back("strength")
			to_show.push_back("enable")
			set_menu_visibility(toolbar_settings.height_list, true)
			to_show.push_back("height_offset")
			to_show.push_back("random_height")
			set_menu_visibility(toolbar_settings.scale_list, true)
			to_show.push_back("fixed_scale")
			to_show.push_back("random_scale")
			set_menu_visibility(toolbar_settings.rotation_list, true)
			to_show.push_back("fixed_spin")
			to_show.push_back("random_spin")
			to_show.push_back("fixed_angle")
			to_show.push_back("random_angle")
			to_show.push_back("align_to_normal")
			set_menu_visibility(toolbar_settings.color_list, true)
			to_show.push_back("vertex_color")
			to_show.push_back("random_darken")
			to_show.push_back("random_hue")

		_:
			pass

	# Advanced menu settings
	to_show.push_back("automatic_regions")
	to_show.push_back("align_to_view")
	to_show.push_back("show_cursor_while_painting")
	to_show.push_back("gamma")
	to_show.push_back("jitter")
	toolbar_settings.show_settings(to_show)

	operation_builder = null
	if p_operation == Terrain3DEditor.GRADIENT:
		operation_builder = GradientOperationBuilder.new()
		operation_builder.tool_settings = toolbar_settings

	if plugin.editor:
		plugin.editor.set_tool(p_tool)
		plugin.editor.set_operation(p_operation)

	_on_setting_changed()
	plugin.update_region_grid()


func _on_setting_changed() -> void:
	if not plugin.asset_dock:
		return
	brush_data = toolbar_settings.get_settings()
	brush_data["asset_id"] = plugin.asset_dock.get_current_list().get_selected_id()
	update_decal()
	plugin.editor.set_brush_data(brush_data)


func update_decal() -> void:
	var mouse_buttons: int = Input.get_mouse_button_mask()
	if not visible or \
			not plugin.terrain or \
			brush_data.is_empty() or \
			mouse_buttons & MOUSE_BUTTON_RIGHT or \
			(mouse_buttons & MOUSE_BUTTON_LEFT and not brush_data["show_cursor_while_painting"]) or \
			plugin.editor.get_tool() == Terrain3DEditor.REGION:
		decal.visible = false
		for gradient_decal in gradient_decals:
			gradient_decal.visible = false
		return
	else:
		# Wait for cursor to recenter after right-click before revealing
		# See https://github.com/godotengine/godot/issues/70098
		await get_tree().create_timer(.05).timeout 
		decal.visible = true

	decal.size = Vector3.ONE * brush_data["size"]
	if brush_data["align_to_view"]:
		var cam: Camera3D = plugin.terrain.get_camera();
		if (cam):
			decal.rotation.y = cam.rotation.y
	else:
		decal.rotation.y = 0

	# Set texture and color
	if picking != Terrain3DEditor.TOOL_MAX:
		decal.texture_albedo = ring_texture
		decal.size = Vector3.ONE * 10. * plugin.terrain.get_mesh_vertex_spacing()
		match picking:
			Terrain3DEditor.HEIGHT:
				decal.modulate = COLOR_PICK_HEIGHT
			Terrain3DEditor.COLOR:
				decal.modulate = COLOR_PICK_COLOR
			Terrain3DEditor.ROUGHNESS:
				decal.modulate = COLOR_PICK_ROUGH
		decal.modulate.a = 1.0
	else:
		decal.texture_albedo = brush_data["brush"][1]
		match plugin.editor.get_tool():
			Terrain3DEditor.HEIGHT:
				match plugin.editor.get_operation():
					Terrain3DEditor.ADD:
						decal.modulate = COLOR_RAISE
					Terrain3DEditor.SUBTRACT:
						decal.modulate = COLOR_LOWER
					Terrain3DEditor.MULTIPLY:
						decal.modulate = COLOR_EXPAND
					Terrain3DEditor.DIVIDE:
						decal.modulate = COLOR_REDUCE
					Terrain3DEditor.REPLACE:
						decal.modulate = COLOR_FLATTEN
					Terrain3DEditor.AVERAGE:
						decal.modulate = COLOR_SMOOTH
					Terrain3DEditor.GRADIENT:
						decal.modulate = COLOR_SLOPE
					_:
						decal.modulate = Color.WHITE
				decal.modulate.a = max(.3, brush_data["strength"] * .01)
			Terrain3DEditor.TEXTURE:
				match plugin.editor.get_operation():
					Terrain3DEditor.REPLACE:
						decal.modulate = COLOR_PAINT
						decal.modulate.a = 1.0
					Terrain3DEditor.ADD:
						decal.modulate = COLOR_SPRAY
						decal.modulate.a = max(.3, brush_data["strength"] * .01)
					_:
						decal.modulate = Color.WHITE
			Terrain3DEditor.COLOR:
				decal.modulate = brush_data["color"].srgb_to_linear()*.5
				decal.modulate.a = max(.3, brush_data["strength"] * .01)
			Terrain3DEditor.ROUGHNESS:
				decal.modulate = COLOR_ROUGHNESS
				decal.modulate.a = max(.3, brush_data["strength"] * .01)
			Terrain3DEditor.AUTOSHADER:
				decal.modulate = COLOR_AUTOSHADER
				decal.modulate.a = 1.0
			Terrain3DEditor.HOLES:
				decal.modulate = COLOR_HOLES
				decal.modulate.a = 1.0
			Terrain3DEditor.NAVIGATION:
				decal.modulate = COLOR_NAVIGATION
				decal.modulate.a = 1.0
			Terrain3DEditor.INSTANCER:
				decal.texture_albedo = ring_texture
				decal.modulate = COLOR_INSTANCER
				decal.modulate.a = 1.0
			_:
				decal.modulate = Color.WHITE
				decal.modulate.a = max(.3, brush_data["strength"] * .01)
	decal.size.y = max(1000, decal.size.y)
	decal.albedo_mix = 1.0
	decal.cull_mask = 1 << ( plugin.terrain.get_mouse_layer() - 1 )
	decal_timer.start()
	
	for gradient_decal in gradient_decals:
		gradient_decal.visible = false
	
	if plugin.editor.get_operation() == Terrain3DEditor.GRADIENT:
		var index := 0
		for point in brush_data["gradient_points"]:
			if point != Vector3.ZERO:
				var point_decal: Decal = _get_gradient_decal(index)
				point_decal.visible = true
				point_decal.position = point
				index += 1


func _get_gradient_decal(index: int) -> Decal:
	if gradient_decals.size() > index:
		return gradient_decals[index]
	
	var gradient_decal := Decal.new()
	gradient_decal = Decal.new()
	gradient_decal.texture_albedo = ring_texture
	gradient_decal.modulate = COLOR_SLOPE
	gradient_decal.size = Vector3.ONE * 10. * plugin.terrain.get_mesh_vertex_spacing()
	gradient_decal.size.y = 1000.
	gradient_decal.cull_mask = decal.cull_mask
	add_child(gradient_decal)
	
	gradient_decals.push_back(gradient_decal)
	return gradient_decal


func set_decal_rotation(p_rot: float) -> void:
	decal.rotation.y = p_rot


func _on_picking(p_type: int, p_callback: Callable) -> void:
	picking = p_type
	picking_callback = p_callback
	update_decal()


func clear_picking() -> void:
	picking = Terrain3DEditor.TOOL_MAX


func is_picking() -> bool:
	if picking != Terrain3DEditor.TOOL_MAX:
		return true
	
	if operation_builder and operation_builder.is_picking():
		return true
	
	return false


func pick(p_global_position: Vector3) -> void:
	if picking != Terrain3DEditor.TOOL_MAX:
		var color: Color
		match picking:
			Terrain3DEditor.HEIGHT:
				color = plugin.terrain.get_storage().get_pixel(Terrain3DStorage.TYPE_HEIGHT, p_global_position)
			Terrain3DEditor.ROUGHNESS:
				color = plugin.terrain.get_storage().get_pixel(Terrain3DStorage.TYPE_COLOR, p_global_position)
			Terrain3DEditor.COLOR:
				color = plugin.terrain.get_storage().get_color(p_global_position)
			Terrain3DEditor.ANGLE:
				color = Color(plugin.terrain.get_storage().get_angle(p_global_position), 0., 0., 1.)
			Terrain3DEditor.SCALE:
				color = Color(plugin.terrain.get_storage().get_scale(p_global_position), 0., 0., 1.)
			_:
				push_error("Unsupported picking type: ", picking)
				return
		picking_callback.call(picking, color, p_global_position)
		picking = Terrain3DEditor.TOOL_MAX
	
	elif operation_builder and operation_builder.is_picking():
		operation_builder.pick(p_global_position, plugin.terrain)

