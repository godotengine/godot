extends PanelContainer


signal resource_changed(resource: Resource, index: int)
signal resource_inspected(resource: Resource)
signal resource_selected

var list: ListContainer
var entries: Array[ListEntry]
var selected_index: int = 0


func _init() -> void:
	list = ListContainer.new()
	
	var root: VBoxContainer = VBoxContainer.new()
	var scroll: ScrollContainer = ScrollContainer.new()
	var label: Label = Label.new()
	
	label.set_text("Textures")
	label.set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER)
	label.set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER)
	scroll.set_v_size_flags(SIZE_EXPAND_FILL)
	scroll.set_h_size_flags(SIZE_EXPAND_FILL)
	list.set_v_size_flags(SIZE_EXPAND_FILL)
	list.set_h_size_flags(SIZE_EXPAND_FILL)
	
	scroll.add_child(list)
	root.add_child(label)
	root.add_child(scroll)
	add_child(root)
	
	set_custom_minimum_size(Vector2(256, 448))

	
func _ready() -> void:
	get_child(0).get_child(0).set("theme_override_styles/normal", get_theme_stylebox("bg", "EditorInspectorCategory"))
	get_child(0).get_child(0).set("theme_override_fonts/font", get_theme_font("bold", "EditorFonts"))
	get_child(0).get_child(0).set("theme_override_font_sizes/font_size",get_theme_font_size("bold_size", "EditorFonts"))
	set("theme_override_styles/panel", get_theme_stylebox("panel", "Panel"))


func clear() -> void:
	for i in entries:
		i.get_parent().remove_child(i)
		i.queue_free()
	entries.clear()


func add_item(p_resource: Resource = null) -> void:
	var entry: ListEntry = ListEntry.new()
	var index: int = entries.size()
	
	entry.set_edited_resource(p_resource)
	entry.selected.connect(set_selected_index.bind(index))
	entry.inspected.connect(notify_resource_inspected)
	entry.changed.connect(notify_resource_changed.bind(index))
	
	if p_resource:
		entry.set_selected(index == selected_index)
		if not p_resource.id_changed.is_connected(set_selected_after_swap):
			p_resource.id_changed.connect(set_selected_after_swap)
	
	list.add_child(entry)
	entries.push_back(entry)


func set_selected_after_swap(p_old_index: int, p_new_index: int) -> void:
	set_selected_index(clamp(p_new_index, 0, entries.size() - 2))


func set_selected_index(p_index: int) -> void:
	selected_index = p_index
	emit_signal("resource_selected")
	
	for i in entries.size():
		var entry: ListEntry = entries[i]
		entry.set_selected(i == selected_index)


func get_selected_index() -> int:
	return selected_index


func notify_resource_inspected(p_resource: Resource) -> void:
	emit_signal("resource_inspected", p_resource)


func notify_resource_changed(p_resource: Resource, p_index: int) -> void:
	emit_signal("resource_changed", p_resource, p_index)
	if !p_resource:
		var last_offset: int = 2
		if p_index == entries.size()-2:
			last_offset = 3
		selected_index = clamp(selected_index, 0, entries.size() - last_offset)	


##############################################################
## class ListContainer
##############################################################

	
class ListContainer extends Container:
	var height: float = 0
	
	func _notification(p_what) -> void:
		if p_what == NOTIFICATION_SORT_CHILDREN:
			height = 0
			var index: int = 0
			var separation: float = 4
			for c in get_children():
				if is_instance_valid(c):
					var width: float = size.x / 3
					c.size = Vector2(width,width) - Vector2(separation, separation)
					c.position = Vector2(index % 3, index / 3) * width + Vector2(separation/3, separation/3)
					height = max(height, c.position.y + width)
					index += 1
					
	func _get_minimum_size() -> Vector2:
		return Vector2(0, height)


##############################################################
## class ListEntry
##############################################################


class ListEntry extends VBoxContainer:
	signal selected()
	signal changed(resource: Terrain3DTexture)
	signal inspected(resource: Terrain3DTexture)
	
	var resource: Terrain3DTexture
	var drop_data: bool = false
	var is_hovered: bool = false
	var is_selected: bool = false
	
	var button_clear: TextureButton
	var button_edit: TextureButton
	var name_label: Label
	
	@onready var add_icon: Texture2D = get_theme_icon("Add", "EditorIcons")
	@onready var clear_icon: Texture2D = get_theme_icon("Close", "EditorIcons")
	@onready var edit_icon: Texture2D = get_theme_icon("Edit", "EditorIcons")
	@onready var background: StyleBox = get_theme_stylebox("pressed", "Button")
	@onready var focus: StyleBox = get_theme_stylebox("focus", "Button")
	

	func _ready() -> void:
		var icon_size: Vector2 = Vector2(12, 12)
		
		button_clear = TextureButton.new()
		button_clear.set_texture_normal(clear_icon)
		button_clear.set_custom_minimum_size(icon_size)
		button_clear.set_h_size_flags(Control.SIZE_SHRINK_END)
		button_clear.set_visible(resource != null)
		button_clear.pressed.connect(clear)
		add_child(button_clear)
		
		button_edit = TextureButton.new()
		button_edit.set_texture_normal(edit_icon)
		button_edit.set_custom_minimum_size(icon_size)
		button_edit.set_h_size_flags(Control.SIZE_SHRINK_END)
		button_edit.set_visible(resource != null)
		button_edit.pressed.connect(edit)
		add_child(button_edit)
		
		name_label = Label.new()
		add_child(name_label, true)
		name_label.visible = false
		name_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		name_label.vertical_alignment = VERTICAL_ALIGNMENT_BOTTOM
		name_label.size_flags_vertical = Control.SIZE_EXPAND_FILL
		name_label.add_theme_color_override("font_shadow_color", Color.BLACK)
		name_label.add_theme_constant_override("shadow_offset_x", 1)
		name_label.add_theme_constant_override("shadow_offset_y", 1)
		name_label.add_theme_font_size_override("font_size", 15)
		name_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		name_label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
		name_label.text = "Add New"
		
		
	func _notification(p_what) -> void:
		match p_what:
			NOTIFICATION_DRAW:
				var rect: Rect2 = Rect2(Vector2.ZERO, get_size())
				if !resource:
					draw_style_box(background, rect)
					draw_texture(add_icon, (get_size() / 2) - (add_icon.get_size() / 2))
				else:
					name_label.text = resource.get_name()
					self_modulate = resource.get_albedo_color()
					var texture: Texture2D = resource.get_albedo_texture()
					if texture:
						draw_texture_rect(texture, rect, false)
						texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST_WITH_MIPMAPS
				name_label.add_theme_font_size_override("font_size", 4 + rect.size.x/10)
				if drop_data:
					draw_style_box(focus, rect)
				if is_hovered:
					draw_rect(rect, Color(1,1,1,0.2))
				if is_selected:
					draw_style_box(focus, rect)
			NOTIFICATION_MOUSE_ENTER:
				is_hovered = true
				name_label.visible = true
				queue_redraw()
			NOTIFICATION_MOUSE_EXIT:
				is_hovered = false
				name_label.visible = false
				drop_data = false
				queue_redraw()

	
	func _gui_input(p_event: InputEvent) -> void:
		if p_event is InputEventMouseButton:
			if p_event.is_pressed():
				match p_event.get_button_index():
					MOUSE_BUTTON_LEFT:
						# If `Add new` is clicked
						if !resource:
							set_edited_resource(Terrain3DTexture.new(), false)
							edit()
						else:
							emit_signal("selected")
					MOUSE_BUTTON_RIGHT:
						if resource:
							edit()
					MOUSE_BUTTON_MIDDLE:
						if resource:
							clear()


	func _can_drop_data(p_at_position: Vector2, p_data: Variant) -> bool:
		drop_data = false
		if typeof(p_data) == TYPE_DICTIONARY:
			if p_data.files.size() == 1:
				queue_redraw()
				drop_data = true
		return drop_data

		
	func _drop_data(p_at_position: Vector2, p_data: Variant) -> void:
		if typeof(p_data) == TYPE_DICTIONARY:
			var res: Resource = load(p_data.files[0])
			if res is Terrain3DTexture:
				set_edited_resource(res, false)
			if res is Texture2D:
				var surf: Terrain3DTexture = Terrain3DTexture.new()
				surf.set_albedo_texture(res)
				set_edited_resource(surf, false)

	
	func set_edited_resource(p_res: Terrain3DTexture, p_no_signal: bool = true) -> void:
		resource = p_res
		if resource:
			resource.setting_changed.connect(_on_texture_changed)
			resource.file_changed.connect(_on_texture_changed)
		
		if button_clear:
			button_clear.set_visible(resource != null)
			
		queue_redraw()
		if !p_no_signal:
			emit_signal("changed", resource)


	func _on_texture_changed() -> void:
		emit_signal("changed", resource)


	func set_selected(value: bool) -> void:
		is_selected = value
		queue_redraw()


	func clear() -> void:
		if resource:
			set_edited_resource(null, false)

	
	func edit() -> void:
		emit_signal("selected")
		emit_signal("inspected", resource)
