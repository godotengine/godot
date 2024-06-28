extends VBoxContainer
	
	
signal tool_changed(p_tool: Terrain3DEditor.Tool, p_operation: Terrain3DEditor.Operation)

const ICON_REGION_ADD: String = "res://addons/terrain_3d/icons/region_add.svg"
const ICON_REGION_REMOVE: String = "res://addons/terrain_3d/icons/region_remove.svg"
const ICON_HEIGHT_ADD: String = "res://addons/terrain_3d/icons/height_add.svg"
const ICON_HEIGHT_SUB: String = "res://addons/terrain_3d/icons/height_sub.svg"
const ICON_HEIGHT_MUL: String = "res://addons/terrain_3d/icons/height_mul.svg"
const ICON_HEIGHT_DIV: String = "res://addons/terrain_3d/icons/height_div.svg"
const ICON_HEIGHT_FLAT: String = "res://addons/terrain_3d/icons/height_flat.svg"
const ICON_HEIGHT_SLOPE: String = "res://addons/terrain_3d/icons/height_slope.svg"
const ICON_HEIGHT_SMOOTH: String = "res://addons/terrain_3d/icons/height_smooth.svg"
const ICON_PAINT_TEXTURE: String = "res://addons/terrain_3d/icons/texture_paint.svg"
const ICON_SPRAY_TEXTURE: String = "res://addons/terrain_3d/icons/texture_spray.svg"
const ICON_COLOR: String = "res://addons/terrain_3d/icons/color_paint.svg"
const ICON_WETNESS: String = "res://addons/terrain_3d/icons/wetness.svg"
const ICON_AUTOSHADER: String = "res://addons/terrain_3d/icons/autoshader.svg"
const ICON_HOLES: String = "res://addons/terrain_3d/icons/holes.svg"
const ICON_NAVIGATION: String = "res://addons/terrain_3d/icons/navigation.svg"
const ICON_INSTANCER: String = "res://addons/terrain_3d/icons/multimesh.svg"

var tool_group: ButtonGroup = ButtonGroup.new()


func _init() -> void:
	set_custom_minimum_size(Vector2(20, 0))


func _ready() -> void:
	tool_group.connect("pressed", _on_tool_selected)
	
	add_tool_button(Terrain3DEditor.REGION, Terrain3DEditor.ADD, "Add Region", load(ICON_REGION_ADD), tool_group)
	add_tool_button(Terrain3DEditor.REGION, Terrain3DEditor.SUBTRACT, "Remove Region", load(ICON_REGION_REMOVE), tool_group)
	add_child(HSeparator.new())
	add_tool_button(Terrain3DEditor.HEIGHT, Terrain3DEditor.ADD, "Raise", load(ICON_HEIGHT_ADD), tool_group)
	add_tool_button(Terrain3DEditor.HEIGHT, Terrain3DEditor.SUBTRACT, "Lower", load(ICON_HEIGHT_SUB), tool_group)
	add_tool_button(Terrain3DEditor.HEIGHT, Terrain3DEditor.MULTIPLY, "Expand (Away from 0)", load(ICON_HEIGHT_MUL), tool_group)
	add_tool_button(Terrain3DEditor.HEIGHT, Terrain3DEditor.DIVIDE, "Reduce (Towards 0)", load(ICON_HEIGHT_DIV), tool_group)
	add_tool_button(Terrain3DEditor.HEIGHT, Terrain3DEditor.REPLACE, "Flatten", load(ICON_HEIGHT_FLAT), tool_group)
	add_tool_button(Terrain3DEditor.HEIGHT, Terrain3DEditor.GRADIENT, "Slope", load(ICON_HEIGHT_SLOPE), tool_group)
	add_tool_button(Terrain3DEditor.HEIGHT, Terrain3DEditor.AVERAGE, "Smooth", load(ICON_HEIGHT_SMOOTH), tool_group)
	add_child(HSeparator.new())
	add_tool_button(Terrain3DEditor.TEXTURE, Terrain3DEditor.REPLACE, "Paint Base Texture", load(ICON_PAINT_TEXTURE), tool_group)
	add_tool_button(Terrain3DEditor.TEXTURE, Terrain3DEditor.ADD, "Spray Overlay Texture", load(ICON_SPRAY_TEXTURE), tool_group)
	add_tool_button(Terrain3DEditor.AUTOSHADER, Terrain3DEditor.REPLACE, "Autoshader", load(ICON_AUTOSHADER), tool_group)
	add_child(HSeparator.new())
	add_tool_button(Terrain3DEditor.COLOR, Terrain3DEditor.REPLACE, "Paint Color", load(ICON_COLOR), tool_group)
	add_tool_button(Terrain3DEditor.ROUGHNESS, Terrain3DEditor.REPLACE, "Paint Wetness", load(ICON_WETNESS), tool_group)
	add_child(HSeparator.new())
	add_tool_button(Terrain3DEditor.HOLES, Terrain3DEditor.REPLACE, "Create Holes", load(ICON_HOLES), tool_group)
	add_tool_button(Terrain3DEditor.NAVIGATION, Terrain3DEditor.REPLACE, "Paint Navigable Area", load(ICON_NAVIGATION), tool_group)
	add_tool_button(Terrain3DEditor.INSTANCER, Terrain3DEditor.ADD, "Instance Meshes", load(ICON_INSTANCER), tool_group)

	var buttons: Array[BaseButton] = tool_group.get_buttons()
	buttons[0].set_pressed(true)


func add_tool_button(p_tool: Terrain3DEditor.Tool, p_operation: Terrain3DEditor.Operation,
		p_tip: String, p_icon: Texture2D, p_group: ButtonGroup) -> void:
		
	var button: Button = Button.new()
	button.set_name(p_tip.to_pascal_case())
	button.set_meta("Tool", p_tool)
	button.set_meta("Operation", p_operation)
	button.set_tooltip_text(p_tip)
	button.set_button_icon(p_icon)
	button.set_button_group(p_group)
	button.set_flat(true)
	button.set_toggle_mode(true)
	button.set_h_size_flags(SIZE_SHRINK_END)
	add_child(button)


func _on_tool_selected(p_button: BaseButton) -> void:
	emit_signal("tool_changed", p_button.get_meta("Tool", -1), p_button.get_meta("Operation", -1))
