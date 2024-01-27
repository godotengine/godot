extends PanelContainer

signal picking(type, callback)
signal setting_changed

enum Layout {
	HORIZONTAL,
	VERTICAL,
	GRID,
}

enum SettingType {
	CHECKBOX,
	SLIDER,
	DOUBLE_SLIDER,
	COLOR_SELECT,
	PICKER,
	POINT_PICKER,
}

const PointPicker: Script = preload("res://addons/terrain_3d/editor/components/point_picker.gd")
const DEFAULT_BRUSH: String = "circle0.exr"
const BRUSH_PATH: String = "res://addons/terrain_3d/editor/brushes"
const PICKER_ICON: String = "res://addons/terrain_3d/icons/icon_picker.svg"

const NONE: int = 0x0
const ALLOW_LARGER: int = 0x1
const ALLOW_SMALLER: int = 0x2
const ALLOW_OUT_OF_BOUNDS: int = 0x3

var brush_preview_material: ShaderMaterial

var list: HBoxContainer
var advanced_list: VBoxContainer
var settings: Dictionary = {}


func _ready() -> void:
	list = HBoxContainer.new()
	add_child(list, true)
	
	add_brushes(list)

	add_setting(SettingType.SLIDER, "size", 50, list, "m", 4, 200, 1, ALLOW_LARGER)
	add_setting(SettingType.SLIDER, "opacity", 10, list, "%", 1, 100)
	add_setting(SettingType.CHECKBOX, "enable", true, list)
	
	add_setting(SettingType.COLOR_SELECT, "color", Color.WHITE, list)
	add_setting(SettingType.PICKER, "color picker", Terrain3DEditor.COLOR, list)
	
	add_setting(SettingType.SLIDER, "roughness", 0, list, "%", -100, 100, 1)
	add_setting(SettingType.PICKER, "roughness picker", Terrain3DEditor.ROUGHNESS, list)
	
	add_setting(SettingType.SLIDER, "height", 50, list, "m", -500, 500, 0.1, ALLOW_OUT_OF_BOUNDS)
	add_setting(SettingType.PICKER, "height picker", Terrain3DEditor.HEIGHT, list)
	add_setting(SettingType.DOUBLE_SLIDER, "slope", 0, list, "°", 0, 180, 1)
	
	add_setting(SettingType.POINT_PICKER, "gradient_points", Terrain3DEditor.HEIGHT, list)
	add_setting(SettingType.CHECKBOX, "drawable", false, list)
	
	settings["drawable"].toggled.connect(_on_drawable_toggled)

	var spacer: Control = Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	list.add_child(spacer, true)

	## Advanced Settings Menu
	advanced_list = create_submenu(list, "Advanced", Layout.VERTICAL)
	add_setting(SettingType.CHECKBOX, "automatic_regions", true, advanced_list)
	add_setting(SettingType.CHECKBOX, "align_to_view", true, advanced_list)
	add_setting(SettingType.CHECKBOX, "show_cursor_while_painting", true, advanced_list)
	advanced_list.add_child(HSeparator.new(), true)
	add_setting(SettingType.SLIDER, "gamma", 1.0, advanced_list, "γ", 0.1, 2.0, 0.01)
	add_setting(SettingType.SLIDER, "jitter", 50, advanced_list, "%", 0, 100)


func create_submenu(p_parent: Control, p_button_name: String, p_layout: Layout) -> Container:
	var menu_button: Button = Button.new()
	menu_button.set_text(p_button_name)
	menu_button.set_toggle_mode(true)
	menu_button.set_v_size_flags(SIZE_SHRINK_CENTER)
	menu_button.connect("toggled", _on_show_submenu.bind(menu_button))

	var submenu: PopupPanel = PopupPanel.new()
	submenu.connect("popup_hide", menu_button.set_pressed_no_signal.bind(false))
	submenu.set("theme_override_styles/panel", get_theme_stylebox("panel", "PopupMenu"))
	
	var sublist: Container
	match(p_layout):
		Layout.GRID:
			sublist = GridContainer.new()
		Layout.VERTICAL:
			sublist = VBoxContainer.new()
		Layout.HORIZONTAL, _:
			sublist = HBoxContainer.new()
	
	p_parent.add_child(menu_button, true)
	menu_button.add_child(submenu, true)
	submenu.add_child(sublist, true)
	
	return sublist


func _on_show_submenu(p_toggled: bool, p_button: Button) -> void:
	var popup: PopupPanel = p_button.get_child(0)
	var popup_pos: Vector2 = p_button.get_screen_transform().origin
	popup.set_visible(p_toggled)
	popup_pos.y -= popup.get_size().y
	popup.set_position(popup_pos)


func add_brushes(p_parent: Control) -> void:
	var brush_list: GridContainer = create_submenu(p_parent, "Brush", Layout.GRID)
	brush_list.name = "BrushList"

	var brush_button_group: ButtonGroup = ButtonGroup.new()
	brush_button_group.connect("pressed", _on_setting_changed)
	var default_brush_btn: Button
	
	var dir: DirAccess = DirAccess.open(BRUSH_PATH)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "":
			if !dir.current_is_dir() and file_name.ends_with(".exr"):
				var img: Image = Image.load_from_file(BRUSH_PATH + "/" + file_name)
				_black_to_alpha(img)
				var tex: ImageTexture = ImageTexture.create_from_image(img)

				var btn: Button = Button.new()
				btn.set_custom_minimum_size(Vector2.ONE * 100)
				btn.set_button_icon(tex)
				btn.set_meta("image", img)
				btn.set_expand_icon(true)
				btn.set_material(_get_brush_preview_material())
				btn.set_toggle_mode(true)
				btn.set_button_group(brush_button_group)
				btn.mouse_entered.connect(_on_brush_hover.bind(true, btn))
				btn.mouse_exited.connect(_on_brush_hover.bind(false, btn))
				brush_list.add_child(btn, true)
				if file_name == DEFAULT_BRUSH:
					default_brush_btn = btn 
				
				var lbl: Label = Label.new()
				btn.add_child(lbl, true)
				lbl.text = file_name.get_basename()
				lbl.visible = false
				lbl.position.y = 70
				lbl.add_theme_color_override("font_shadow_color", Color.BLACK)
				lbl.add_theme_constant_override("shadow_offset_x", 1)
				lbl.add_theme_constant_override("shadow_offset_y", 1)
				lbl.add_theme_font_size_override("font_size", 16)
				
			file_name = dir.get_next()
	
	brush_list.columns = sqrt(brush_list.get_child_count()) + 2
	
	if not default_brush_btn:
		default_brush_btn = brush_button_group.get_buttons()[0]
	default_brush_btn.set_pressed(true)
	
	settings["brush"] = brush_button_group

	# Optionally erase the main brush button text and replace it with the texture
#	var select_brush_btn: Button = brush_list.get_parent().get_parent()
#	select_brush_btn.set_button_icon(default_brush_btn.get_button_icon())
#	select_brush_btn.set_custom_minimum_size(Vector2.ONE * 36)
#	select_brush_btn.set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER)
#	select_brush_btn.set_expand_icon(true)


func _on_brush_hover(p_hovering: bool, p_button: Button) -> void:
	if p_button.get_child_count() > 0:
		var child = p_button.get_child(0)
		if child is Label:
			if p_hovering:
				child.visible = true
			else:
				child.visible = false


func _on_pick(p_type: Terrain3DEditor.Tool) -> void:
	emit_signal("picking", p_type, _on_picked)


func _on_picked(p_type: Terrain3DEditor.Tool, p_color: Color, p_global_position: Vector3) -> void:
	match p_type:
		Terrain3DEditor.HEIGHT:
			settings["height"].value = p_color.r
		Terrain3DEditor.COLOR:
			settings["color"].color = p_color
		Terrain3DEditor.ROUGHNESS:
			# 200... -.5 converts 0,1 to -100,100
			settings["roughness"].value = round(200 * (p_color.a - 0.5))
	_on_setting_changed()


func _on_point_pick(p_type: Terrain3DEditor.Tool, p_name: String) -> void:
	assert(p_type == Terrain3DEditor.HEIGHT)
	emit_signal("picking", p_type, _on_point_picked.bind(p_name))


func _on_point_picked(p_type: Terrain3DEditor.Tool, p_color: Color, p_global_position: Vector3, p_name: String) -> void:
	assert(p_type == Terrain3DEditor.HEIGHT)
	
	var point: Vector3 = p_global_position
	point.y = p_color.r
	settings[p_name].add_point(point)
	_on_setting_changed()


func add_setting(p_type: SettingType, p_name: StringName, p_value: Variant, p_parent: Control, 
		p_suffix: String = "", p_min_value: float = 0.0, p_max_value: float = 0.0, p_step: float = 1.0,
		p_flags: int = NONE) -> void:

	var container: HBoxContainer = HBoxContainer.new()
	var label: Label = Label.new()
	var control: Control

	container.set_v_size_flags(SIZE_EXPAND_FILL)

	match p_type:
		SettingType.SLIDER, SettingType.DOUBLE_SLIDER:
			label.set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER)
			label.set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER)
			label.set_custom_minimum_size(Vector2(32, 0))
			label.set_v_size_flags(SIZE_SHRINK_CENTER)
			label.set_text(p_name.capitalize() + ": ")
			container.add_child(label, true)
			
			var slider: Control
			if p_type == SettingType.SLIDER:
				control = EditorSpinSlider.new()
				control.set_flat(true)
				control.set_hide_slider(true)
				control.connect("value_changed", _on_setting_changed)
				control.set_max(p_max_value)
				control.set_min(p_min_value)
				control.set_step(p_step)
				control.set_value(p_value)
				control.set_suffix(p_suffix)
				control.set_v_size_flags(SIZE_SHRINK_CENTER)
			
				slider = HSlider.new()
				slider.share(control)
				if p_flags & ALLOW_LARGER:
					slider.set_allow_greater(true)
				if p_flags & ALLOW_SMALLER:
					slider.set_allow_lesser(true)
					
			else:
				control = Label.new()
				control.set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER)
				control.set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER)
				slider = DoubleSlider.new()
				slider.label = control
				slider.suffix = p_suffix
				slider.connect("value_changed", _on_setting_changed)
			
			control.set_custom_minimum_size(Vector2(75, 0))
			slider.set_max(p_max_value)
			slider.set_min(p_min_value)
			slider.set_step(p_step)
			slider.set_value(p_value)
			slider.set_v_size_flags(SIZE_SHRINK_CENTER)
			slider.set_h_size_flags(SIZE_SHRINK_END | SIZE_EXPAND)
			slider.set_custom_minimum_size(Vector2(100, 10))
			
			container.add_child(slider, true)
				
		SettingType.CHECKBOX:
			control = CheckBox.new()
			control.set_text(p_name.capitalize())
			control.set_pressed_no_signal(p_value)
			control.connect("pressed", _on_setting_changed)
			
		SettingType.COLOR_SELECT:
			control = ColorPickerButton.new()
			control.set_custom_minimum_size(Vector2(100, 10))
			control.color = Color.WHITE
			control.edit_alpha = false
			control.get_picker().set_color_mode(ColorPicker.MODE_HSV)
			control.connect("color_changed", _on_setting_changed)
			
		SettingType.PICKER:
			control = Button.new()
			control.icon = load(PICKER_ICON)
			control.tooltip_text = "Pick value from the Terrain"
			control.connect("pressed", _on_pick.bind(p_value))
		
		SettingType.POINT_PICKER:
			control = PointPicker.new()
			control.connect("pressed", _on_point_pick.bind(p_value, p_name))
			control.connect("value_changed", _on_setting_changed)
		
	container.add_child(control, true)
	p_parent.add_child(container, true)
	
	settings[p_name] = control


func get_setting(p_setting: String) -> Variant:
	var object: Object = settings[p_setting]
	var value: Variant
	if object is Range:
		value = object.get_value()
	elif object is DoubleSlider:
		value = [object.get_min_value(), object.get_max_value()]
	elif object is ButtonGroup:
		var img: Image = object.get_pressed_button().get_meta("image")
		var tex: Texture2D = object.get_pressed_button().get_button_icon()
		value = [ img, tex ]
	elif object is CheckBox:
		value = object.is_pressed()
	elif object is ColorPickerButton:
		value = object.color
	elif object is PointPicker:
		value = object.get_points()
	return value


func hide_settings(p_settings: PackedStringArray) -> void:
	for setting in settings.keys():
		var object: Object = settings[setting]
		if object is Control:
			object.get_parent().show()
	
	for setting in p_settings:
		if settings.has(setting):
			var object: Object = settings[setting]
			if object is Control:
				object.get_parent().hide()


func _on_setting_changed(p_data: Variant = null) -> void:
	# If a button was clicked on a submenu
	if p_data is Button and p_data.get_parent().get_parent() is PopupPanel:
		if p_data.get_parent().name == "BrushList":
			# Optionally Set selected brush texture in main brush button
#			p_data.get_parent().get_parent().get_parent().set_button_icon(p_data.get_button_icon())
			# Hide popup
			p_data.get_parent().get_parent().set_visible(false)
			# Hide label
			if p_data.get_child_count() > 0:
				p_data.get_child(0).visible = false

	emit_signal("setting_changed")
	

func _on_drawable_toggled(p_button_pressed: bool) -> void:
	if not p_button_pressed:
		settings["gradient_points"].clear()


func _get_brush_preview_material() -> ShaderMaterial:
	if !brush_preview_material:
		brush_preview_material = ShaderMaterial.new()
		
		var shader: Shader = Shader.new()
		var code: String = "shader_type canvas_item;\n"
		
		code += "varying vec4 v_vertex_color;\n"
		code += "void vertex() {\n"
		code += "	v_vertex_color = COLOR;\n"
		code += "}\n"
		code += "void fragment(){\n"
		code += "	vec4 tex = texture(TEXTURE, UV);\n"
		code += "	COLOR.a *= pow(tex.r, 0.666);\n"
		code += "	COLOR.rgb = v_vertex_color.rgb;\n"
		code += "}\n"
		
		shader.set_code(code)
		
		brush_preview_material.set_shader(shader)
		
	return brush_preview_material


func _black_to_alpha(p_image: Image) -> void:
	if p_image.get_format() != Image.FORMAT_RGBAF:
		p_image.convert(Image.FORMAT_RGBAF)

	for y in p_image.get_height():
		for x in p_image.get_width():
			var color: Color = p_image.get_pixel(x,y)
			color.a = color.get_luminance()
			p_image.set_pixel(x, y, color)


#### Sub Class DoubleSlider

class DoubleSlider extends Range:
	
	var label: Label
	var suffix: String
	var grabbed: bool = false
	var _max_value: float
	
	
	func _gui_input(p_event: InputEvent) -> void:
		if p_event is InputEventMouseButton:
			if p_event.get_button_index() == MOUSE_BUTTON_LEFT:
				grabbed = p_event.is_pressed()
				set_min_max(p_event.get_position().x)
				
		if p_event is InputEventMouseMotion:
			if grabbed:
				set_min_max(p_event.get_position().x)
		
		
	func _notification(p_what: int) -> void:
		if p_what == NOTIFICATION_RESIZED:
			pass
		if p_what == NOTIFICATION_DRAW:
			var bg: StyleBox = get_theme_stylebox("slider", "HSlider")
			var bg_height: float = bg.get_minimum_size().y
			draw_style_box(bg, Rect2(Vector2(0, (size.y - bg_height) / 2), Vector2(size.x, bg_height)))
			
			var grabber: Texture2D = get_theme_icon("grabber", "HSlider")
			var area: StyleBox = get_theme_stylebox("grabber_area", "HSlider")
			var h: float = size.y / 2 - grabber.get_size().y / 2
			
			var minpos: Vector2 = Vector2((min_value / _max_value) * size.x - grabber.get_size().x / 2, h)
			var maxpos: Vector2 = Vector2((max_value / _max_value) * size.x - grabber.get_size().x / 2, h)
			
			draw_style_box(area, Rect2(Vector2(minpos.x + grabber.get_size().x / 2, (size.y - bg_height) / 2), Vector2(maxpos.x - minpos.x, bg_height)))
			
			draw_texture(grabber, minpos)
			draw_texture(grabber, maxpos)
			
			
	func set_max(p_value: float) -> void:
		max_value = p_value
		if _max_value == 0:
			_max_value = max_value
		update_label()
		
		
	func set_min_max(p_xpos: float) -> void:
		var mid_value_normalized: float = ((max_value + min_value) / 2.0) / _max_value
		var mid_value: float = size.x * mid_value_normalized
		var min_active: bool = p_xpos < mid_value
		var xpos_ranged: float = snappedf((p_xpos / size.x) * _max_value, step)
		
		if min_active:
			min_value = xpos_ranged
		else:
			max_value = xpos_ranged
		
		min_value = clamp(min_value, 0, max_value - 10)
		max_value = clamp(max_value, min_value + 10, _max_value)
		
		update_label()
		emit_signal("setting_changed", value)
		queue_redraw()
		
		
	func update_label() -> void:
		if label:
			label.set_text(str(min_value) + suffix + "/" + str(max_value) + suffix)
