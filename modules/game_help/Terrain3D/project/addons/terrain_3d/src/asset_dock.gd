@tool
extends PanelContainer
#class_name Terrain3DAssetDock

signal confirmation_closed
signal confirmation_confirmed
signal confirmation_canceled

const PS_DOCK_SLOT: String = "terrain3d/config/dock_slot"
const PS_DOCK_TILE_SIZE: String = "terrain3d/config/dock_tile_size"
const PS_DOCK_FLOATING: String = "terrain3d/config/dock_floating"
const PS_DOCK_PINNED: String = "terrain3d/config/dock_always_on_top"
const PS_DOCK_WINDOW_POSITION: String = "terrain3d/config/dock_window_position"
const PS_DOCK_WINDOW_SIZE: String = "terrain3d/config/dock_window_size"

var texture_list: ListContainer
var mesh_list: ListContainer
var _current_list: ListContainer
var _last_thumb_update_time: int = 0
const MAX_UPDATE_TIME: int = 1000

var placement_opt: OptionButton
var floating_btn: Button
var pinned_btn: Button
var size_slider: HSlider
var box: BoxContainer
var buttons: BoxContainer
var textures_btn: Button
var meshes_btn: Button
var asset_container: ScrollContainer
var confirm_dialog: ConfirmationDialog
var _confirmed: bool = false

# Used only for editor, so change to single visible/hiddden
enum {
	HIDDEN = -1,
	SIDEBAR = 0,
	BOTTOM = 1,
	WINDOWED = 2,
}
var state: int = HIDDEN

var window: Window
var _godot_editor_window: Window # The main Godot Editor window
var _godot_last_state: Window.Mode = Window.MODE_FULLSCREEN

enum {
	POS_LEFT_UL = 0,
	POS_LEFT_BL = 1,
	POS_LEFT_UR = 2,
	POS_LEFT_BR = 3,
	POS_RIGHT_UL = 4,
	POS_RIGHT_BL = 5,
	POS_RIGHT_UR = 6,
	POS_RIGHT_BR = 7,
	POS_BOTTOM = 8,
	POS_MAX = 9,
}
var slot: int = POS_RIGHT_BR
var _initialized: bool = false
var plugin: EditorPlugin
var editor_settings: EditorSettings


func initialize(p_plugin: EditorPlugin) -> void:
	if p_plugin:
		plugin = p_plugin

	# Get editor window. Structure is root:Window/EditorNode/Base Control
	_godot_editor_window = plugin.get_editor_interface().get_base_control().get_parent().get_parent()
	_godot_last_state = _godot_editor_window.mode
	
	placement_opt = $Box/Buttons/PlacementOpt
	pinned_btn = $Box/Buttons/Pinned
	floating_btn = $Box/Buttons/Floating
	floating_btn.owner = null
	size_slider = $Box/Buttons/SizeSlider
	size_slider.owner = null
	box = $Box
	buttons = $Box/Buttons
	textures_btn = $Box/Buttons/TexturesBtn
	meshes_btn = $Box/Buttons/MeshesBtn
	asset_container = $Box/ScrollContainer

	texture_list = ListContainer.new()
	texture_list.plugin = plugin
	texture_list.type = Terrain3DAssets.TYPE_TEXTURE
	asset_container.add_child(texture_list)
	mesh_list = ListContainer.new()
	mesh_list.plugin = plugin
	mesh_list.type = Terrain3DAssets.TYPE_MESH
	mesh_list.visible = false
	asset_container.add_child(mesh_list)
	_current_list = texture_list

	editor_settings = EditorInterface.get_editor_settings()
	load_editor_settings()

	# Connect signals
	resized.connect(update_layout)
	textures_btn.pressed.connect(_on_textures_pressed)
	meshes_btn.pressed.connect(_on_meshes_pressed)
	placement_opt.item_selected.connect(set_slot)
	floating_btn.pressed.connect(make_dock_float)
	pinned_btn.toggled.connect(_on_pin_changed)
	pinned_btn.visible = false
	size_slider.value_changed.connect(_on_slider_changed)
	plugin.ui.toolbar.tool_changed.connect(_on_tool_changed)

	meshes_btn.add_theme_font_size_override("font_size", 16 * EditorInterface.get_editor_scale())
	textures_btn.add_theme_font_size_override("font_size", 16 * EditorInterface.get_editor_scale())

	_initialized = true
	update_dock(plugin.visible)
	update_layout()


func _ready() -> void:
	if not _initialized:
		return
		
	# Setup styles
	set("theme_override_styles/panel", get_theme_stylebox("panel", "Panel"))
	# Avoid saving icon resources in tscn when editing w/ a tool script
	if plugin.get_editor_interface().get_edited_scene_root() != self:
		pinned_btn.icon = get_theme_icon("Pin", "EditorIcons")
		pinned_btn.text = ""
		floating_btn.icon = get_theme_icon("MakeFloating", "EditorIcons")
		floating_btn.text = ""

	update_thumbnails()
	confirm_dialog = ConfirmationDialog.new()
	add_child(confirm_dialog)
	confirm_dialog.hide()
	confirm_dialog.confirmed.connect(func(): _confirmed = true; \
		emit_signal("confirmation_closed"); \
		emit_signal("confirmation_confirmed") )
	confirm_dialog.canceled.connect(func(): _confirmed = false; \
		emit_signal("confirmation_closed"); \
		emit_signal("confirmation_canceled") )


func get_current_list() -> ListContainer:
	return _current_list


## Dock placement

func set_slot(p_slot: int) -> void:
	p_slot = clamp(p_slot, 0, POS_MAX-1)
	
	if slot != p_slot:
		slot = p_slot
		placement_opt.selected = slot
		save_editor_settings()
		plugin.select_terrain()
		update_dock(plugin.visible)


func remove_dock(p_force: bool = false) -> void:
	if state == SIDEBAR:
		plugin.remove_control_from_docks(self)
		state = HIDDEN

	elif state == BOTTOM:
		plugin.remove_control_from_bottom_panel(self)
		state = HIDDEN

	# If windowed and destination is not window or final exit, otherwise leave
	elif state == WINDOWED and p_force:
		if not window:
			return
		var parent: Node = get_parent()
		if parent:
			parent.remove_child(self)
			_godot_editor_window.mouse_entered.disconnect(_on_godot_window_entered)
			_godot_editor_window.focus_entered.disconnect(_on_godot_focus_entered)
			_godot_editor_window.focus_exited.disconnect(_on_godot_focus_exited)
			window.hide()
			window.queue_free()
			window = null
		floating_btn.button_pressed = false
		floating_btn.visible = true
		pinned_btn.visible = false
		placement_opt.visible = true
		state = HIDDEN
		update_dock(plugin.visible) # return window to side/bottom


func update_dock(p_visible: bool) -> void:
	update_assets()
	if not _initialized:
		return

	if window:
		return
	elif floating_btn.button_pressed:
		# No window, but floating button pressed, occurs when from editor settings
		make_dock_float()
		return

	remove_dock()
	# Add dock to new destination
	# Sidebar
	if slot < POS_BOTTOM:
		state = SIDEBAR
		plugin.add_control_to_dock(slot, self)
	elif slot == POS_BOTTOM:
		state = BOTTOM
		plugin.add_control_to_bottom_panel(self, "Terrain3D")
		if p_visible:
			plugin.make_bottom_panel_item_visible(self)
		

func update_layout() -> void:
	if not _initialized:
		return

	# Detect if we have a new window from Make floating, grab it so we can free it properly
	if not window and get_parent() and get_parent().get_parent() is Window:
		window = get_parent().get_parent()
		make_dock_float()
		return # Will call this function again upon display

	var size_parent: Control = size_slider.get_parent()
	# Vertical layout in window / sidebar
	if window or slot < POS_BOTTOM:
		box.vertical = true
		buttons.vertical = false

		if size.x >= 500 and size_parent != buttons:
			size_slider.reparent(buttons)
			buttons.move_child(size_slider, 3)
		elif size.x < 500 and size_parent != box:
			size_slider.reparent(box)
			box.move_child(size_slider, 1)
		floating_btn.reparent(buttons)
		buttons.move_child(floating_btn, 4)

	# Wide layout on bottom bar
	else:
		size_slider.reparent(buttons)
		buttons.move_child(size_slider, 3)
		floating_btn.reparent(box)
		box.vertical = false
		buttons.vertical = true

	save_editor_settings()


func update_thumbnails() -> void:
	if not is_instance_valid(plugin.terrain):
		return
	if _current_list.type == Terrain3DAssets.TYPE_MESH and \
			Time.get_ticks_msec() - _last_thumb_update_time > MAX_UPDATE_TIME:
		plugin.terrain.assets.create_mesh_thumbnails()
		_last_thumb_update_time = Time.get_ticks_msec()
		for mesh_asset in mesh_list.entries:
			mesh_asset.queue_redraw()
## Dock Button handlers


func _on_pin_changed(toggled: bool) -> void:
	if window:
		window.always_on_top = pinned_btn.button_pressed
	save_editor_settings()


func _on_slider_changed(value: float) -> void:
	if texture_list:
		texture_list.set_entry_width(value)
	if mesh_list:
		mesh_list.set_entry_width(value)
	save_editor_settings()


func _on_textures_pressed() -> void:
	_current_list = texture_list
	texture_list.update_asset_list()
	texture_list.visible = true
	mesh_list.visible = false
	textures_btn.button_pressed = true
	meshes_btn.button_pressed = false
	texture_list.set_selected_id(texture_list.selected_id)
	plugin.get_editor_interface().edit_node(plugin.terrain)


func _on_meshes_pressed() -> void:
	_current_list = mesh_list
	mesh_list.update_asset_list()
	mesh_list.visible = true
	texture_list.visible = false
	meshes_btn.button_pressed = true
	textures_btn.button_pressed = false
	mesh_list.set_selected_id(mesh_list.selected_id)
	plugin.get_editor_interface().edit_node(plugin.terrain)
	update_thumbnails()


func _on_tool_changed(p_tool: Terrain3DEditor.Tool, p_operation: Terrain3DEditor.Operation) -> void:
	if p_tool == Terrain3DEditor.INSTANCER:
		_on_meshes_pressed()
	elif p_tool == Terrain3DEditor.TEXTURE:
		_on_textures_pressed()


## Update Dock Contents


func update_assets() -> void:
	if not _initialized:
		return
	
	# Verify signals to individual lists
	if plugin.is_terrain_valid() and plugin.terrain.assets:
		if not plugin.terrain.assets.textures_changed.is_connected(texture_list.update_asset_list):
			plugin.terrain.assets.textures_changed.connect(texture_list.update_asset_list)
		if not plugin.terrain.assets.meshes_changed.is_connected(mesh_list.update_asset_list):
			plugin.terrain.assets.meshes_changed.connect(mesh_list.update_asset_list)

	_current_list.update_asset_list()

## Window Management


func make_dock_float() -> void:
	# If already created (eg from editor Make Floating)	
	if not window:
		remove_dock()
		create_window()

	state = WINDOWED
	pinned_btn.visible = true
	floating_btn.visible = false
	placement_opt.visible = false
	window.title = "Terrain3D Asset Dock"
	window.always_on_top = pinned_btn.button_pressed
	window.close_requested.connect(remove_dock.bind(true))
	visible = true # Is hidden when pops off of bottom. ??
	_godot_editor_window.grab_focus()


func create_window() -> void:
	window = Window.new()
	window.wrap_controls = true
	var mc := MarginContainer.new()
	mc.set_anchors_preset(PRESET_FULL_RECT, false)
	mc.add_child(self)
	window.add_child(mc)
	window.set_transient(false)
	window.set_size(get_setting(PS_DOCK_WINDOW_SIZE, Vector2i(512, 512)))
	window.set_position(get_setting(PS_DOCK_WINDOW_POSITION, Vector2i(704, 284)))
	plugin.add_child(window)
	window.show()
	window.window_input.connect(_on_window_input)
	window.focus_exited.connect(_on_window_focus_exited)
	_godot_editor_window.mouse_entered.connect(_on_godot_window_entered)
	_godot_editor_window.focus_entered.connect(_on_godot_focus_entered)
	_godot_editor_window.focus_exited.connect(_on_godot_focus_exited)


func _on_window_input(event: InputEvent) -> void:
	# Capture CTRL+S when doc focused to save scene)
	if event is InputEventKey and event.keycode == KEY_S and event.pressed and event.is_command_or_control_pressed():
		save_editor_settings()
		plugin.get_editor_interface().save_scene()


func _on_window_focus_exited() -> void:
	# Capture window position w/o other changes
	save_editor_settings()


func _on_godot_window_entered() -> void:
	if is_instance_valid(window) and window.has_focus():
		_godot_editor_window.grab_focus()


func _on_godot_focus_entered() -> void:
	# If asset dock is windowed, and Godot was minimized, and now is not, restore asset dock window
	if is_instance_valid(window):
		if _godot_last_state == Window.MODE_MINIMIZED and _godot_editor_window.mode != Window.MODE_MINIMIZED:
			window.show()
			_godot_last_state = _godot_editor_window.mode
			_godot_editor_window.grab_focus()


func _on_godot_focus_exited() -> void:
	if is_instance_valid(window) and _godot_editor_window.mode == Window.MODE_MINIMIZED:
		window.hide()
		_godot_last_state = _godot_editor_window.mode


## Manage Editor Settings


func get_setting(p_str: String, p_default: Variant) -> Variant:
	if editor_settings.has_setting(p_str):
		return editor_settings.get_setting(p_str)
	else:
		return p_default


func load_editor_settings() -> void:
	floating_btn.button_pressed = get_setting(PS_DOCK_FLOATING, false)
	pinned_btn.button_pressed = get_setting(PS_DOCK_PINNED, true)
	size_slider.value = get_setting(PS_DOCK_TILE_SIZE, 83)
	set_slot(get_setting(PS_DOCK_SLOT, POS_BOTTOM))
	_on_slider_changed(size_slider.value)
	# Window pos/size set on window creation in update_dock
	update_dock(plugin.visible)
	
	
func save_editor_settings() -> void:
	if not _initialized:
		return
	editor_settings.set_setting(PS_DOCK_SLOT, slot)
	editor_settings.set_setting(PS_DOCK_TILE_SIZE, size_slider.value)
	editor_settings.set_setting(PS_DOCK_FLOATING, floating_btn.button_pressed)
	editor_settings.set_setting(PS_DOCK_PINNED, pinned_btn.button_pressed)
	if window:
		editor_settings.set_setting(PS_DOCK_WINDOW_SIZE, window.size)
		editor_settings.set_setting(PS_DOCK_WINDOW_POSITION, window.position)


##############################################################
## class ListContainer
##############################################################

	
class ListContainer extends Container:
	var plugin: EditorPlugin
	var type := Terrain3DAssets.TYPE_TEXTURE
	var entries: Array[ListEntry]
	var selected_id: int = 0
	var height: float = 0
	var width: float = 83
	var focus_style: StyleBox

	
	func _ready() -> void:
		set_v_size_flags(SIZE_EXPAND_FILL)
		set_h_size_flags(SIZE_EXPAND_FILL)
		focus_style = get_theme_stylebox("focus", "Button").duplicate()
		focus_style.set_border_width_all(2)
		focus_style.set_border_color(Color(1, 1, 1, .67))


	func clear() -> void:
		for e in entries:
			e.get_parent().remove_child(e)
			e.queue_free()
		entries.clear()


	func update_asset_list() -> void:
		clear()
		
		# Grab terrain
		var t: Terrain3D
		if plugin.is_terrain_valid():
			t = plugin.terrain
		elif is_instance_valid(plugin._last_terrain) and plugin.is_terrain_valid(plugin._last_terrain):
			t = plugin._last_terrain
		else:
			return
		
		if not t.assets:
			return
		
		if type == Terrain3DAssets.TYPE_TEXTURE:
			var texture_count: int = t.assets.get_texture_count()
			for i in texture_count:
				var texture: Terrain3DTextureAsset = t.assets.get_texture(i)
				add_item(texture)
			if texture_count < Terrain3DAssets.MAX_TEXTURES:
				add_item()
		else:
			var mesh_count: int = t.assets.get_mesh_count()
			for i in mesh_count:
				var mesh: Terrain3DMeshAsset = t.assets.get_mesh_asset(i)
				add_item(mesh, t.assets)
			if mesh_count < Terrain3DAssets.MAX_MESHES:
				add_item()
			if selected_id >= mesh_count or selected_id < 0:
				set_selected_id(0)


	func add_item(p_resource: Resource = null, p_assets: Terrain3DAssets = null) -> void:
		var entry: ListEntry = ListEntry.new()
		entry.focus_style = focus_style
		var id: int = entries.size()
		
		entry.set_edited_resource(p_resource)
		entry.hovered.connect(_on_resource_hovered.bind(id))
		entry.selected.connect(set_selected_id.bind(id))
		entry.inspected.connect(_on_resource_inspected)
		entry.changed.connect(_on_resource_changed.bind(id))
		entry.type = type
		entry.asset_list = p_assets
		add_child(entry)
		entries.push_back(entry)
		
		if p_resource:
			entry.set_selected(id == selected_id)
			if not p_resource.id_changed.is_connected(set_selected_after_swap):
				p_resource.id_changed.connect(set_selected_after_swap)


	func _on_resource_hovered(p_id: int):
		if type == Terrain3DAssets.TYPE_MESH:
			if plugin.terrain:
				plugin.terrain.assets.create_mesh_thumbnails(p_id)

	
	func set_selected_after_swap(p_type: Terrain3DAssets.AssetType, p_old_id: int, p_new_id: int) -> void:
		set_selected_id(clamp(p_new_id, 0, entries.size() - 2))


	func set_selected_id(p_id: int) -> void:
		selected_id = p_id
		
		for i in entries.size():
			var entry: ListEntry = entries[i]
			entry.set_selected(i == selected_id)
		
		plugin.select_terrain()

		# Select Paint tool if clicking a texture
		if type == Terrain3DAssets.TYPE_TEXTURE and plugin.editor.get_tool() != Terrain3DEditor.TEXTURE:
			var paint_btn: Button = plugin.ui.toolbar.get_node_or_null("PaintBaseTexture")
			if paint_btn:
				paint_btn.set_pressed(true)
				plugin.ui._on_tool_changed(Terrain3DEditor.TEXTURE, Terrain3DEditor.REPLACE)

		elif type == Terrain3DAssets.TYPE_MESH and plugin.editor.get_tool() != Terrain3DEditor.INSTANCER:
			var instancer_btn: Button = plugin.ui.toolbar.get_node_or_null("InstanceMeshes")
			if instancer_btn:
				instancer_btn.set_pressed(true)
				plugin.ui._on_tool_changed(Terrain3DEditor.INSTANCER, Terrain3DEditor.ADD)
		
		# Update editor with selected brush
		plugin.ui._on_setting_changed()


	func _on_resource_inspected(p_resource: Resource) -> void:
		await get_tree().create_timer(.01).timeout
		plugin.get_editor_interface().edit_resource(p_resource)
	
	
	func _on_resource_changed(p_resource: Resource, p_id: int) -> void:
		if not p_resource:
			var asset_dock: Control = get_parent().get_parent().get_parent()
			if type == Terrain3DAssets.TYPE_TEXTURE:
				asset_dock.confirm_dialog.dialog_text = "Are you sure you want to clear this texture?"
			else:
				asset_dock.confirm_dialog.dialog_text = "Are you sure you want to clear this mesh and delete all instances?"
			asset_dock.confirm_dialog.popup_centered()
			await asset_dock.confirmation_closed
			if not asset_dock._confirmed:
				update_asset_list()
				return
			
		if not plugin.is_terrain_valid():
			plugin.select_terrain()
			await get_tree().create_timer(.01).timeout

		if plugin.is_terrain_valid():
			if type == Terrain3DAssets.TYPE_TEXTURE:
				plugin.terrain.get_assets().set_texture(p_id, p_resource)
			else:
				plugin.terrain.get_assets().set_mesh_asset(p_id, p_resource)
				await get_tree().create_timer(.01).timeout
				plugin.terrain.assets.create_mesh_thumbnails(p_id)

			# If removing an entry, clear inspector
			if not p_resource:
				plugin.get_editor_interface().inspect_object(null)			
				
		# If null resource, remove last 
		if not p_resource:
			var last_offset: int = 2
			if p_id == entries.size()-2:
				last_offset = 3
			set_selected_id(clamp(selected_id, 0, entries.size() - last_offset))

		# Update editor with selected brush
		plugin.ui._on_setting_changed()


	func get_selected_id() -> int:
		return selected_id



	func set_entry_width(value: float) -> void:
		width = clamp(value, 56, 230)
		redraw()


	func get_entry_width() -> float:
		return width
	

	func redraw() -> void:
		height = 0
		var id: int = 0
		var separation: float = 4
		var columns: int = 3
		columns = clamp(size.x / width, 1, 100)
		
		for c in get_children():
			if is_instance_valid(c):
				c.size = Vector2(width, width) - Vector2(separation, separation)
				c.position = Vector2(id % columns, id / columns) * width + \
					Vector2(separation / columns, separation / columns)
				height = max(height, c.position.y + width)
				id += 1


	# Needed to enable ScrollContainer scroll bar
	func _get_minimum_size() -> Vector2:
		return Vector2(0, height)

		
	func _notification(p_what) -> void:
		if p_what == NOTIFICATION_SORT_CHILDREN:
			redraw()


##############################################################
## class ListEntry
##############################################################


class ListEntry extends VBoxContainer:
	signal hovered()
	signal selected()
	signal changed(resource: Resource)
	signal inspected(resource: Resource)
	
	var resource: Resource
	var type := Terrain3DAssets.TYPE_TEXTURE
	var _thumbnail: Texture2D
	var drop_data: bool = false
	var is_hovered: bool = false
	var is_selected: bool = false
	var asset_list: Terrain3DAssets
	
	var button_clear: TextureButton
	var button_edit: TextureButton
	var name_label: Label
	
	@onready var add_icon: Texture2D = get_theme_icon("Add", "EditorIcons")
	@onready var clear_icon: Texture2D = get_theme_icon("Close", "EditorIcons")
	@onready var edit_icon: Texture2D = get_theme_icon("Edit", "EditorIcons")
	@onready var background: StyleBox = get_theme_stylebox("pressed", "Button")
	var focus_style: StyleBox


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
		if type == Terrain3DAssets.TYPE_TEXTURE:
			name_label.text = "Add Texture"
		else:
			name_label.text = "Add Mesh"

		
	func _notification(p_what) -> void:
		match p_what:
			NOTIFICATION_DRAW:
				var rect: Rect2 = Rect2(Vector2.ZERO, get_size())
				if !resource:
					draw_style_box(background, rect)
					draw_texture(add_icon, (get_size() / 2) - (add_icon.get_size() / 2))
				else:
					if type == Terrain3DAssets.TYPE_TEXTURE:
						name_label.text = (resource as Terrain3DTextureAsset).get_name()
						self_modulate = resource.get_albedo_color()
						_thumbnail = resource.get_albedo_texture()
						if _thumbnail:
							draw_texture_rect(_thumbnail, rect, false)
							texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST_WITH_MIPMAPS
					else:
						name_label.text = (resource as Terrain3DMeshAsset).get_name()
						var id: int = (resource as Terrain3DMeshAsset).get_id()
						_thumbnail = resource.get_thumbnail()
						if _thumbnail:
							draw_texture_rect(_thumbnail, rect, false)
							texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
						else:
							draw_rect(rect, Color(.15, .15, .15, 1.))
				name_label.add_theme_font_size_override("font_size", 4 + rect.size.x/10)
				if drop_data:
					draw_style_box(focus_style, rect)
				if is_hovered:
					draw_rect(rect, Color(1, 1, 1, 0.2))
				if is_selected:
					draw_style_box(focus_style, rect)
			NOTIFICATION_MOUSE_ENTER:
				is_hovered = true
				name_label.visible = true
				emit_signal("hovered")
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
							if type == Terrain3DAssets.TYPE_TEXTURE:
								set_edited_resource(Terrain3DTextureAsset.new(), false)
							else:
								set_edited_resource(Terrain3DMeshAsset.new(), false)
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
			if res is Texture2D and type == Terrain3DAssets.TYPE_TEXTURE:
				var ta := Terrain3DTextureAsset.new()
				if resource is Terrain3DTextureAsset:
					ta.id = resource.id
				ta.set_albedo_texture(res)
				set_edited_resource(ta, false)
				resource = ta
			elif res is Terrain3DTextureAsset and type == Terrain3DAssets.TYPE_TEXTURE:
				if resource is Terrain3DTextureAsset:
					res.id = resource.id
				set_edited_resource(res, false)
			elif res is PackedScene and type == Terrain3DAssets.TYPE_MESH:
				var ma := Terrain3DMeshAsset.new()
				if resource is Terrain3DMeshAsset:
					ma.id = resource.id
				ma.set_scene_file(res)
				set_edited_resource(ma, false)
				resource = ma
			elif res is Terrain3DMeshAsset and type == Terrain3DAssets.TYPE_MESH:
				if resource is Terrain3DMeshAsset:
					res.id = resource.id
				set_edited_resource(res, false)
			emit_signal("selected")
			emit_signal("inspected", resource)



	func set_edited_resource(p_res: Resource, p_no_signal: bool = true) -> void:
		resource = p_res
		if resource:
			resource.setting_changed.connect(_on_resource_changed)
			resource.file_changed.connect(_on_resource_changed)
		
		if button_clear:
			button_clear.set_visible(resource != null)
			
		queue_redraw()
		if !p_no_signal:
			emit_signal("changed", resource)


	func _on_resource_changed() -> void:
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
