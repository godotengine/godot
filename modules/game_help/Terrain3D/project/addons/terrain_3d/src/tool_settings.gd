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
	COLOR_SELECT,
	DOUBLE_SLIDER,
	OPTION,
	PICKER,
	MULTI_PICKER,
	SLIDER,
	TYPE_MAX,
}

const MultiPicker: Script = preload("res://addons/terrain_3d/src/multi_picker.gd")
const DEFAULT_BRUSH: String = "circle0.exr"
const BRUSH_PATH: String = "res://addons/terrain_3d/brushes"
const PICKER_ICON: String = "res://addons/terrain_3d/icons/picker.svg"

# Add settings flags
const NONE: int = 0x0
const ALLOW_LARGER: int = 0x1
const ALLOW_SMALLER: int = 0x2
const ALLOW_OUT_OF_BOUNDS: int = 0x3 # LARGER|SMALLER
const NO_LABEL: int = 0x4
const ADD_SEPARATOR: int = 0x8
const ADD_SPACER: int = 0x10

var brush_preview_material: ShaderMaterial
var select_brush_button: Button

var main_list: HBoxContainer
var advanced_list: VBoxContainer
var height_list: VBoxContainer
var scale_list: VBoxContainer
var rotation_list: VBoxContainer
var color_list: VBoxContainer
var settings: Dictionary = {}


func _ready() -> void:
	main_list = HBoxContainer.new()
	add_child(main_list, true)
	
	## Common Settings
	add_brushes(main_list)

	add_setting({ "name":"size", "type":SettingType.SLIDER, "list":main_list, "default":50, "unit":"m",
								"range":Vector3(2, 200, 1), "flags":ALLOW_LARGER|ADD_SPACER })
		
	add_setting({ "name":"strength", "type":SettingType.SLIDER, "list":main_list, "default":10, 
								"unit":"%", "range":Vector3(1, 100, 1), "flags":ALLOW_LARGER })

	add_setting({ "name":"enable", "type":SettingType.CHECKBOX, "list":main_list, "default":true })

	add_setting({ "name":"height", "type":SettingType.SLIDER, "list":main_list, "default":50, 
								"unit":"m", "range":Vector3(-500, 500, 0.1), "flags":ALLOW_OUT_OF_BOUNDS })
	add_setting({ "name":"height_picker", "type":SettingType.PICKER, "list":main_list, 
								"default":Terrain3DEditor.HEIGHT, "flags":NO_LABEL })
	
	add_setting({ "name":"color", "type":SettingType.COLOR_SELECT, "list":main_list, 
								"default":Color.WHITE, "flags":ADD_SEPARATOR })
	add_setting({ "name":"color_picker", "type":SettingType.PICKER, "list":main_list, 
								"default":Terrain3DEditor.COLOR, "flags":NO_LABEL })

	add_setting({ "name":"roughness", "type":SettingType.SLIDER, "list":main_list, "default":0,
								"unit":"%", "range":Vector3(-100, 100, 1), "flags":ADD_SEPARATOR })
	add_setting({ "name":"roughness_picker", "type":SettingType.PICKER, "list":main_list, 
								"default":Terrain3DEditor.ROUGHNESS, "flags":NO_LABEL })

	add_setting({ "name":"enable_texture", "label":"Texture", "type":SettingType.CHECKBOX, 
								"list":main_list, "default":true })

	add_setting({ "name":"enable_angle", "label":"Angle", "type":SettingType.CHECKBOX, 
								"list":main_list, "default":true, "flags":ADD_SEPARATOR })
	add_setting({ "name":"angle", "type":SettingType.SLIDER, "list":main_list, "default":0,
								"unit":"%", "range":Vector3(0, 337.5, 22.5), "flags":NO_LABEL })
	add_setting({ "name":"angle_picker", "type":SettingType.PICKER, "list":main_list, 
								"default":Terrain3DEditor.ANGLE, "flags":NO_LABEL })
	add_setting({ "name":"dynamic_angle", "label":"Dynamic", "type":SettingType.CHECKBOX, 
								"list":main_list, "default":false, "flags":ADD_SPACER })
	
	add_setting({ "name":"enable_scale", "label":"Scale ±", "type":SettingType.CHECKBOX, 
								"list":main_list, "default":true, "flags":ADD_SEPARATOR })
	add_setting({ "name":"scale", "label":"±", "type":SettingType.SLIDER, "list":main_list, "default":0,
								"unit":"%", "range":Vector3(-60, 80, 20), "flags":NO_LABEL })
	add_setting({ "name":"scale_picker", "type":SettingType.PICKER, "list":main_list, 
								"default":Terrain3DEditor.SCALE, "flags":NO_LABEL })

	## Slope
	add_setting({ "name":"slope", "type":SettingType.DOUBLE_SLIDER, "list":main_list, 
								"default":0, "unit":"°", "range":Vector3(0, 180, 1) })
	add_setting({ "name":"gradient_points", "type":SettingType.MULTI_PICKER, "label":"Points", 
								"list":main_list, "default":Terrain3DEditor.HEIGHT, "flags":ADD_SEPARATOR })
	add_setting({ "name":"drawable", "type":SettingType.CHECKBOX, "list":main_list, "default":false, 
								"flags":ADD_SEPARATOR })
	settings["drawable"].toggled.connect(_on_drawable_toggled)
	
	## Instancer
	height_list = create_submenu(main_list, "Height", Layout.VERTICAL)
	add_setting({ "name":"height_offset", "type":SettingType.SLIDER, "list":height_list, "default":0, 
								"unit":"m", "range":Vector3(-10, 10, 0.05), "flags":ALLOW_OUT_OF_BOUNDS })
	add_setting({ "name":"random_height", "label":"Random Height ±", "type":SettingType.SLIDER, 
								"list":height_list, "default":0, "unit":"m", "range":Vector3(0, 10, 0.05),
								"flags":ALLOW_OUT_OF_BOUNDS })

	scale_list = create_submenu(main_list, "Scale", Layout.VERTICAL)
	add_setting({ "name":"fixed_scale", "type":SettingType.SLIDER, "list":scale_list, "default":100, 
								"unit":"%", "range":Vector3(1, 1000, 1), "flags":ALLOW_OUT_OF_BOUNDS })
	add_setting({ "name":"random_scale", "label":"Random Scale ±", "type":SettingType.SLIDER, "list":scale_list, 
								"default":20, "unit":"%", "range":Vector3(0, 99, 1), "flags":ALLOW_OUT_OF_BOUNDS })

	rotation_list = create_submenu(main_list, "Rotation", Layout.VERTICAL)
	add_setting({ "name":"fixed_spin", "label":"Fixed Spin (Around Y)", "type":SettingType.SLIDER, "list":rotation_list, 
								"default":0, "unit":"°", "range":Vector3(0, 360, 1) })
	add_setting({ "name":"random_spin", "type":SettingType.SLIDER, "list":rotation_list, "default":360, 
								"unit":"°", "range":Vector3(0, 360, 1) })
	add_setting({ "name":"fixed_angle", "label":"Fixed Angle (From Y)", "type":SettingType.SLIDER, "list":rotation_list, 
								"default":0, "unit":"°", "range":Vector3(-85, 85, 1), "flags":ALLOW_OUT_OF_BOUNDS })
	add_setting({ "name":"random_angle", "label":"Random Angle ±", "type":SettingType.SLIDER, "list":rotation_list, 
								"default":10, "unit":"°", "range":Vector3(0, 85, 1), "flags":ALLOW_OUT_OF_BOUNDS })
	add_setting({ "name":"align_to_normal", "type":SettingType.CHECKBOX, "list":rotation_list, "default":false })
	
	color_list = create_submenu(main_list, "Color", Layout.VERTICAL)
	add_setting({ "name":"vertex_color", "type":SettingType.COLOR_SELECT, "list":color_list, 
								"default":Color.WHITE })
	add_setting({ "name":"random_hue", "label":"Random Hue Shift ±", "type":SettingType.SLIDER, 
								"list":color_list, "default":0, "unit":"°", "range":Vector3(0, 360, 1) })
	add_setting({ "name":"random_darken", "type":SettingType.SLIDER, "list":color_list, "default":50, 
								"unit":"%", "range":Vector3(0, 100, 1) })
	#add_setting({ "name":"blend_mode", "type":SettingType.OPTION, "list":color_list, "default":0, 
								#"range":Vector3(0, 3, 1) })

	var spacer: Control = Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	main_list.add_child(spacer, true)

	## Advanced Settings Menu
	advanced_list = create_submenu(main_list, "Advanced", Layout.VERTICAL)
	add_setting({ "name":"automatic_regions", "type":SettingType.CHECKBOX, "list":advanced_list, 
								"default":true })
	add_setting({ "name":"align_to_view", "type":SettingType.CHECKBOX, "list":advanced_list, 
								"default":true })
	add_setting({ "name":"show_cursor_while_painting", "type":SettingType.CHECKBOX, "list":advanced_list, 
								"default":true })
	advanced_list.add_child(HSeparator.new(), true)
	add_setting({ "name":"gamma", "type":SettingType.SLIDER, "list":advanced_list, "default":1.0, 
								"unit":"γ", "range":Vector3(0.1, 2.0, 0.01) })
	add_setting({ "name":"jitter", "type":SettingType.SLIDER, "list":advanced_list, "default":50, 
								"unit":"%", "range":Vector3(0, 100, 1) })


func create_submenu(p_parent: Control, p_button_name: String, p_layout: Layout) -> Container:
	var menu_button: Button = Button.new()
	menu_button.set_text(p_button_name)
	menu_button.set_toggle_mode(true)
	menu_button.set_v_size_flags(SIZE_SHRINK_CENTER)
	menu_button.toggled.connect(_on_show_submenu.bind(menu_button))
	
	var submenu: PopupPanel = PopupPanel.new()
	submenu.popup_hide.connect(menu_button.set_pressed_no_signal.bind(false))
	var panel_style: StyleBox = get_theme_stylebox("panel", "PopupMenu").duplicate()
	panel_style.set_content_margin_all(10)
	submenu.set("theme_override_styles/panel", panel_style)
	submenu.add_to_group("terrain3d_submenus")

	# Pop up menu on hover, hide on exit
	menu_button.mouse_entered.connect(_on_show_submenu.bind(true, menu_button))
	submenu.mouse_exited.connect(_on_show_submenu.bind(false, menu_button))
	
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
	# Don't show if mouse already down (from painting)
	if p_toggled and Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		return
	
	# Hide menu if mouse is not in button or panel 
	var button_rect: Rect2 = Rect2(p_button.get_screen_transform().origin, p_button.get_global_rect().size)
	var in_button: bool = button_rect.has_point(DisplayServer.mouse_get_position())
	var panel: PopupPanel = p_button.get_child(0)
	var panel_rect: Rect2 = Rect2(panel.position, panel.size)
	var in_panel: bool = panel_rect.has_point(DisplayServer.mouse_get_position())
	if not p_toggled and ( in_button or in_panel ):
		return
	
	# Hide all submenus before possibly enabling the current one
	get_tree().call_group("terrain3d_submenus", "set_visible", false)
	var popup: PopupPanel = p_button.get_child(0)
	var popup_pos: Vector2 = p_button.get_screen_transform().origin
	popup.set_visible(p_toggled)
	popup_pos.y -= popup.get_size().y
	popup.set_position(popup_pos)


func add_brushes(p_parent: Control) -> void:
	var brush_list: GridContainer = create_submenu(p_parent, "Brush", Layout.GRID)
	brush_list.name = "BrushList"

	var brush_button_group: ButtonGroup = ButtonGroup.new()
	brush_button_group.pressed.connect(_on_setting_changed)
	var default_brush_btn: Button
	
	var dir: DirAccess = DirAccess.open(BRUSH_PATH)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "":
			if !dir.current_is_dir() and file_name.ends_with(".exr"):
				var img: Image = Image.load_from_file(BRUSH_PATH + "/" + file_name)
				img = Terrain3DUtil.black_to_alpha(img)
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
				btn.name = file_name.get_basename().to_pascal_case()
				btn.add_child(lbl, true)
				lbl.text = btn.name
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

	select_brush_button = brush_list.get_parent().get_parent()
	# Optionally erase the main brush button text and replace it with the texture
#	select_brush_button.set_button_icon(default_brush_btn.get_button_icon())
#	select_brush_button.set_custom_minimum_size(Vector2.ONE * 36)
#	select_brush_button.set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER)
#	select_brush_button.set_expand_icon(true)


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
			settings["height"].value = p_color.r if not is_nan(p_color.r) else 0
		Terrain3DEditor.COLOR:
			settings["color"].color = p_color if not is_nan(p_color.r) else Color.WHITE
		Terrain3DEditor.ROUGHNESS:
			# 200... -.5 converts 0,1 to -100,100
			settings["roughness"].value = round(200 * (p_color.a - 0.5)) if not is_nan(p_color.r) else 0.499
		Terrain3DEditor.ANGLE:
			settings["angle"].value = p_color.r
		Terrain3DEditor.SCALE:
			settings["scale"].value = p_color.r
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


func add_setting(p_args: Dictionary) -> void:
	var p_name: StringName = p_args.get("name", "")
	var p_label: String = p_args.get("label", "") # Optional replacement for name
	var p_type: SettingType = p_args.get("type", SettingType.TYPE_MAX)
	var p_list: Control = p_args.get("list")
	var p_default: Variant = p_args.get("default")
	var p_suffix: String = p_args.get("unit", "")
	var p_range: Vector3 = p_args.get("range", Vector3(0, 0, 1))
	var p_minimum: float = p_range.x
	var p_maximum: float = p_range.y
	var p_step: float = p_range.z
	var p_flags: int = p_args.get("flags", NONE)
	
	if p_name.is_empty() or p_type == SettingType.TYPE_MAX:
		return

	var container: HBoxContainer = HBoxContainer.new()
	container.set_v_size_flags(SIZE_EXPAND_FILL)
	var control: Control	# Houses the setting to be saved
	var pending_children: Array[Control]

	match p_type:
		SettingType.CHECKBOX:
			var checkbox := CheckBox.new()
			checkbox.set_pressed_no_signal(p_default)
			checkbox.pressed.connect(_on_setting_changed)
			pending_children.push_back(checkbox)
			control = checkbox
			
		SettingType.COLOR_SELECT:
			var picker := ColorPickerButton.new()
			picker.set_custom_minimum_size(Vector2(100, 25))
			picker.color = Color.WHITE
			picker.edit_alpha = false
			picker.get_picker().set_color_mode(ColorPicker.MODE_HSV)
			picker.color_changed.connect(_on_setting_changed)
			var popup: PopupPanel = picker.get_popup()
			popup.mouse_exited.connect(Callable(func(p): p.hide()).bind(popup))
			pending_children.push_back(picker)
			control = picker

		SettingType.PICKER:
			var button := Button.new()
			button.set_v_size_flags(SIZE_SHRINK_CENTER)
			button.icon = load(PICKER_ICON)
			button.tooltip_text = "Pick value from the Terrain"
			button.pressed.connect(_on_pick.bind(p_default))
			pending_children.push_back(button)
			control = button

		SettingType.MULTI_PICKER:
			var multi_picker: HBoxContainer = MultiPicker.new()
			multi_picker.pressed.connect(_on_point_pick.bind(p_default, p_name))
			multi_picker.value_changed.connect(_on_setting_changed)
			pending_children.push_back(multi_picker)
			control = multi_picker

		SettingType.OPTION:
			var option := OptionButton.new()
			for i in int(p_maximum):
				option.add_item("a", i)
			option.selected = p_minimum
			option.item_selected.connect(_on_setting_changed)
			pending_children.push_back(option)
			control = option

		SettingType.SLIDER, SettingType.DOUBLE_SLIDER:			
			var slider: Control
			if p_type == SettingType.SLIDER:
				# Create an editable value box
				var spin_slider := EditorSpinSlider.new()
				spin_slider.set_flat(false)
				spin_slider.set_hide_slider(true)
				spin_slider.value_changed.connect(_on_setting_changed)
				spin_slider.set_max(p_maximum)
				spin_slider.set_min(p_minimum)
				spin_slider.set_step(p_step)
				spin_slider.set_value(p_default)
				spin_slider.set_suffix(p_suffix)
				spin_slider.set_v_size_flags(SIZE_SHRINK_CENTER)
				spin_slider.set_custom_minimum_size(Vector2(75, 0))

				# Create horizontal slider linked to the above box
				slider = HSlider.new()
				slider.share(spin_slider)
				if p_flags & ALLOW_LARGER:
					slider.set_allow_greater(true)
				if p_flags & ALLOW_SMALLER:
					slider.set_allow_lesser(true)
				pending_children.push_back(slider)
				pending_children.push_back(spin_slider)
				control = spin_slider
					
			else: # DOUBLE_SLIDER
				var label := Label.new()
				label.set_custom_minimum_size(Vector2(75, 0))
				slider = DoubleSlider.new()
				slider.label = label
				slider.suffix = p_suffix
				slider.setting_changed.connect(_on_setting_changed)
				pending_children.push_back(slider)
				pending_children.push_back(label)
				control = slider
			
			slider.set_max(p_maximum)
			slider.set_min(p_minimum)
			slider.set_step(p_step)
			slider.set_value(p_default)
			slider.set_v_size_flags(SIZE_SHRINK_CENTER)
			slider.set_custom_minimum_size(Vector2(60, 10))

	control.name = p_name.to_pascal_case()
	settings[p_name] = control

	# Setup button labels
	if not (p_flags & NO_LABEL):
		# Labels are actually buttons styled to look like labels
		var label := Button.new()
		label.set("theme_override_styles/normal", get_theme_stylebox("normal", "Label"))
		label.set("theme_override_styles/hover", get_theme_stylebox("normal", "Label"))
		label.set("theme_override_styles/pressed", get_theme_stylebox("normal", "Label"))
		label.set("theme_override_styles/focus", get_theme_stylebox("normal", "Label"))
		label.pressed.connect(_on_label_pressed.bind(p_name, p_default))
		if p_label.is_empty():
			label.set_text(p_name.capitalize() + ": ")
		else:
			label.set_text(p_label.capitalize() + ": ")
		pending_children.push_front(label)

	# Add separators to front
	if p_flags & ADD_SEPARATOR:
		pending_children.push_front(VSeparator.new())
	if p_flags & ADD_SPACER:
		var spacer := Control.new()
		spacer.set_custom_minimum_size(Vector2(5, 0))
		pending_children.push_front(spacer)

	# Add all children to container and list
	for child in pending_children:
		container.add_child(child, true)
	p_list.add_child(container, true)


# If label button is pressed, reset value to default or toggle checkbox
func _on_label_pressed(p_name: String, p_default: Variant) -> void:
	var control: Control = settings.get(p_name)
	if not control:
		return
	if control is CheckBox:
		set_setting(p_name, !control.button_pressed)
	elif p_default != null:
		set_setting(p_name, p_default)


func get_settings() -> Dictionary:
	var dict: Dictionary
	for key in settings.keys():
		dict[key] = get_setting(key)
	return dict


func get_setting(p_setting: String) -> Variant:
	var object: Object = settings.get(p_setting)
	var value: Variant
	if object is Range:
		value = object.get_value()
		# Adjust widths of all sliders on update of values
		var digits: float = count_digits(value)
		var width: float = clamp( (1 + count_digits(value)) * 19., 50, 80) * clamp(EditorInterface.get_editor_scale(), .9, 2)
		object.set_custom_minimum_size(Vector2(width, 0))
	elif object is DoubleSlider:
		value = Vector2(object.get_min_value(), object.get_max_value())
	elif object is ButtonGroup:
		var img: Image = object.get_pressed_button().get_meta("image")
		var tex: Texture2D = object.get_pressed_button().get_button_icon()
		value = [ img, tex ]
	elif object is CheckBox:
		value = object.is_pressed()
	elif object is ColorPickerButton:
		value = object.color
	elif object is MultiPicker:
		value = object.get_points()
	if value == null:
		value = 0
	return value


func set_setting(p_setting: String, p_value: Variant) -> void:
	var object: Object = settings.get(p_setting)
	if object is Range:
		object.set_value(p_value)
	elif object is DoubleSlider: # Expects p_value is Vector2
		object.set_min_value(p_value.x)
		object.set_max_value(p_value.y)
	elif object is ButtonGroup: # Expects p_value is Array [ "button name", boolean ]
		if p_value is Array and p_value.size() == 2:
			for button in object.get_buttons():
				if button.name == p_value[0]:
					button.button_pressed = p_value[1]
	elif object is CheckBox:
		object.button_pressed = p_value
	elif object is ColorPickerButton:
		object.color = p_value
	elif object is MultiPicker: # Expects p_value is PackedVector3Array
		object.points = p_value
	_on_setting_changed(object)


func show_settings(p_settings: PackedStringArray) -> void:
	for setting in settings.keys():
		var object: Object = settings[setting]
		if object is Control:
			if setting in p_settings:
				object.get_parent().show()
			else:
				object.get_parent().hide()
	if select_brush_button:
		if not "brush" in p_settings:
			select_brush_button.hide()
		else:
			select_brush_button.show()


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



# Counts digits of a number including negative sign, decimal points, and up to 3 decimals 
func count_digits(p_value: float) -> int:
	var count: int = 1
	for i in range(5, 0, -1):
		if abs(p_value) >= pow(10, i):
			count = i+1
			break
	if p_value - floor(p_value) >= .1:
		count += 1 # For the decimal
		if p_value*10 - floor(p_value*10.) >= .1: 
			count += 1
			if p_value*100 - floor(p_value*100.) >= .1: 
				count += 1
				if p_value*1000 - floor(p_value*1000.) >= .1: 
					count += 1
	# Negative sign
	if p_value < 0:
		count += 1
	return count
	

#### Sub Class DoubleSlider

class DoubleSlider extends Range:
	signal setting_changed(Vector2)
	var label: Label
	var suffix: String
	var grabbed: bool = false
	var _max_value: float
	# TODO Needs to clamp min and max values. Currently allows max slider to go negative.
	
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
		emit_signal("setting_changed", Vector2(min_value, max_value))
		queue_redraw()
		
		
	func update_label() -> void:
		if label:
			label.set_text(str(min_value) + suffix + "/" + str(max_value) + suffix)
