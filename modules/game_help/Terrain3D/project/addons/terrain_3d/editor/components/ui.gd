extends Node
#class_name Terrain3DUI Cannot be named until Godot #75388


# Includes
const Toolbar: Script = preload("res://addons/terrain_3d/editor/components/toolbar.gd")
const ToolSettings: Script = preload("res://addons/terrain_3d/editor/components/tool_settings.gd")
const TerrainTools: Script = preload("res://addons/terrain_3d/editor/components/terrain_tools.gd")
const RING1: String = "res://addons/terrain_3d/editor/brushes/ring1.exr"
const COLOR_RAISE := Color.WHITE
const COLOR_LOWER := Color.BLACK
const COLOR_SMOOTH := Color(0.5, 0, .1)
const COLOR_EXPAND := Color.ORANGE
const COLOR_REDUCE := Color.BLUE_VIOLET
const COLOR_FLATTEN := Color(0., 0.32, .4)
const COLOR_PAINT := Color.FOREST_GREEN
const COLOR_SPRAY := Color.SEA_GREEN
const COLOR_ROUGHNESS := Color.ROYAL_BLUE
const COLOR_AUTOSHADER := Color.DODGER_BLUE
const COLOR_HOLES := Color.BLACK
const COLOR_NAVIGATION := Color.REBECCA_PURPLE
const COLOR_PICK_COLOR := Color.WHITE
const COLOR_PICK_HEIGHT := Color.DARK_RED
const COLOR_PICK_ROUGH := Color.ROYAL_BLUE


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
var brush_data: Dictionary
@onready var picker_texture: ImageTexture =  ImageTexture.create_from_image(Image.load_from_file(RING1))


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

	decal = Decal.new()
	add_child(decal)
	decal_timer = Timer.new()
	decal_timer.wait_time = .5
	decal_timer.timeout.connect(Callable(func(n):
		if n:
			get_tree().create_tween().tween_property(n, "albedo_mix", 0.0, 0.15)).bind(decal))
	add_child(decal_timer)


func _exit_tree() -> void:
	plugin.remove_control_from_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_SIDE_LEFT, toolbar)
	plugin.remove_control_from_container(EditorPlugin.CONTAINER_SPATIAL_EDITOR_BOTTOM, toolbar_settings)
	toolbar.queue_free()
	toolbar_settings.queue_free()
	terrain_tools.queue_free()
	decal.queue_free()
	decal_timer.queue_free()


func set_visible(p_visible: bool) -> void:
	visible = p_visible
	toolbar.set_visible(p_visible and plugin.terrain)
	terrain_tools.set_visible(p_visible)
	
	if p_visible and plugin.terrain:
		p_visible = plugin.editor.get_tool() != Terrain3DEditor.REGION
	toolbar_settings.set_visible(p_visible and plugin.terrain)
	update_decal()


func _on_tool_changed(p_tool: Terrain3DEditor.Tool, p_operation: Terrain3DEditor.Operation) -> void:
	if not visible or not plugin.terrain:
		return

	if plugin.editor:
		plugin.editor.set_tool(p_tool)
		plugin.editor.set_operation(p_operation)
	
	if p_tool != Terrain3DEditor.REGION:
		# Select which settings to hide. Options:
		# size, opactiy, height, slope, color, roughness, (height|color|roughness) picker
		var to_hide: PackedStringArray = []
		
		if p_tool == Terrain3DEditor.HEIGHT:
			to_hide.push_back("color")
			to_hide.push_back("color picker")
			to_hide.push_back("roughness")
			to_hide.push_back("roughness picker")
			to_hide.push_back("slope")
			to_hide.push_back("enable")
			if p_operation != Terrain3DEditor.REPLACE:
				to_hide.push_back("height")
				to_hide.push_back("height picker")

		elif p_tool == Terrain3DEditor.TEXTURE:
			to_hide.push_back("height")
			to_hide.push_back("height picker")
			to_hide.push_back("color")
			to_hide.push_back("color picker")
			to_hide.push_back("roughness")
			to_hide.push_back("roughness picker")
			to_hide.push_back("slope")
			to_hide.push_back("enable")
			if p_operation == Terrain3DEditor.REPLACE:
				to_hide.push_back("opacity")

		elif p_tool == Terrain3DEditor.COLOR:
			to_hide.push_back("height")
			to_hide.push_back("height picker")
			to_hide.push_back("roughness")
			to_hide.push_back("roughness picker")
			to_hide.push_back("slope")
			to_hide.push_back("enable")

		elif p_tool == Terrain3DEditor.ROUGHNESS:
			to_hide.push_back("height")
			to_hide.push_back("height picker")
			to_hide.push_back("color")
			to_hide.push_back("color picker")
			to_hide.push_back("slope")
			to_hide.push_back("enable")
	
		elif p_tool in [ Terrain3DEditor.AUTOSHADER, Terrain3DEditor.HOLES, Terrain3DEditor.NAVIGATION ]:
			to_hide.push_back("height")
			to_hide.push_back("height picker")
			to_hide.push_back("color")
			to_hide.push_back("color picker")
			to_hide.push_back("roughness")
			to_hide.push_back("roughness picker")
			to_hide.push_back("slope")
			to_hide.push_back("opacity")

		toolbar_settings.hide_settings(to_hide)

	toolbar_settings.set_visible(p_tool != Terrain3DEditor.REGION)	
	_on_setting_changed()
	plugin.update_region_grid()


func _on_setting_changed() -> void:
	if not visible or not plugin.terrain:
		return
	brush_data = {
		"size": int(toolbar_settings.get_setting("size")),
		"opacity": toolbar_settings.get_setting("opacity") / 100.0,
		"height": toolbar_settings.get_setting("height"),
		"texture_index": plugin.texture_dock.get_selected_index(),
		"color": toolbar_settings.get_setting("color"),
		"roughness": toolbar_settings.get_setting("roughness"),
		"enable": toolbar_settings.get_setting("enable"),
		"automatic_regions": toolbar_settings.get_setting("automatic_regions"),
		"align_to_view": toolbar_settings.get_setting("align_to_view"),
		"show_cursor_while_painting": toolbar_settings.get_setting("show_cursor_while_painting"),
		"gamma": toolbar_settings.get_setting("gamma"),
		"jitter": toolbar_settings.get_setting("jitter"),
	}
	var brush_imgs: Array = toolbar_settings.get_setting("brush")
	brush_data["image"] = brush_imgs[0]
	brush_data["texture"] = brush_imgs[1]
	
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
		decal.texture_albedo = picker_texture
		decal.size = Vector3.ONE*10.
		match picking:
			Terrain3DEditor.HEIGHT:
				decal.modulate = COLOR_PICK_HEIGHT
			Terrain3DEditor.COLOR:
				decal.modulate = COLOR_PICK_COLOR
			Terrain3DEditor.ROUGHNESS:
				decal.modulate = COLOR_PICK_ROUGH
		decal.modulate.a = 1.0
	else:
		decal.texture_albedo = brush_data["texture"]		
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
					_:
						decal.modulate = Color.WHITE
				decal.modulate.a = max(.3, brush_data["opacity"])
			Terrain3DEditor.TEXTURE:
				match plugin.editor.get_operation():
					Terrain3DEditor.REPLACE:
						decal.modulate = COLOR_PAINT
						decal.modulate.a = 1.0
					Terrain3DEditor.ADD:
						decal.modulate = COLOR_SPRAY
						decal.modulate.a = max(.3, brush_data["opacity"])
					_:
						decal.modulate = Color.WHITE
			Terrain3DEditor.COLOR:
				decal.modulate = brush_data["color"].srgb_to_linear()*.5
				decal.modulate.a = max(.3, brush_data["opacity"])
			Terrain3DEditor.ROUGHNESS:
				decal.modulate = COLOR_ROUGHNESS
				decal.modulate.a = max(.3, brush_data["opacity"])
			Terrain3DEditor.AUTOSHADER:
				decal.modulate = COLOR_AUTOSHADER
				decal.modulate.a = 1.0
			Terrain3DEditor.HOLES:
				decal.modulate = COLOR_HOLES
				decal.modulate.a = 1.0
			Terrain3DEditor.NAVIGATION:
				decal.modulate = COLOR_NAVIGATION
				decal.modulate.a = 1.0
			_:
				decal.modulate = Color.WHITE
				decal.modulate.a = max(.3, brush_data["opacity"])
	decal.albedo_mix = 1.0
	decal_timer.start()


func set_decal_rotation(p_rot: float) -> void:
	decal.rotation.y = p_rot


func _on_picking(p_type: int, p_callback: Callable) -> void:
	picking = p_type
	picking_callback = p_callback
	update_decal()
