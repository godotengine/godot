extends Object

const WINDOW_SCENE: String = "res://addons/terrain_3d/src/channel_packer.tscn"
const TEMPLATE_PATH: String = "res://addons/terrain_3d/src/channel_packer_import_template.txt"

enum { 
  IMAGE_ALBEDO,
  IMAGE_HEIGHT,
  IMAGE_NORMAL,
  IMAGE_ROUGHNESS,
}

var plugin: EditorPlugin
var editor_interface: EditorInterface
var dialog: AcceptDialog
var save_file_dialog: FileDialog
var open_file_dialog: FileDialog
var invert_green_checkbox: CheckBox
var last_opened_directory: String
var last_saved_directory: String
var packing_albedo: bool = false
var queue_pack_normal_roughness: bool = false
var images: Array[Image] = [null, null, null, null]
var status_label: Label
var no_op: Callable = func(): pass
var last_file_selected_fn: Callable = no_op


func pack_textures_popup() -> void:
	if dialog != null:
		print("Terrain3DChannelPacker: Cannot open pack tool, dialog already open.")
		return

	dialog = (load(WINDOW_SCENE) as PackedScene).instantiate()
	dialog.confirmed.connect(_on_close_requested)
	dialog.canceled.connect(_on_close_requested)
	status_label = dialog.find_child("StatusLabel")
	invert_green_checkbox = dialog.find_child("InvertGreenChannelCheckBox")

	editor_interface = plugin.get_editor_interface()
	_init_file_dialogs()	
	editor_interface.popup_dialog_centered(dialog)

	_init_texture_picker(dialog.find_child("AlbedoVBox"), IMAGE_ALBEDO)
	_init_texture_picker(dialog.find_child("HeightVBox"), IMAGE_HEIGHT)
	_init_texture_picker(dialog.find_child("NormalVBox"), IMAGE_NORMAL)
	_init_texture_picker(dialog.find_child("RoughnessVBox"), IMAGE_ROUGHNESS)
	var pack_button_path: String = "Panel/MarginContainer/VBoxContainer/PackButton" 
	(dialog.get_node(pack_button_path) as Button).pressed.connect(_on_pack_button_pressed)


func _on_close_requested() -> void:
	last_file_selected_fn = no_op
	images = [null, null, null, null]
	dialog.queue_free()
	dialog = null


func _init_file_dialogs() -> void:
	save_file_dialog = FileDialog.new()
	save_file_dialog.set_filters(PackedStringArray(["*.png"]))
	save_file_dialog.set_file_mode(FileDialog.FILE_MODE_SAVE_FILE)
	save_file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	save_file_dialog.file_selected.connect(_on_save_file_selected)

	open_file_dialog = FileDialog.new()
	open_file_dialog.set_filters(PackedStringArray(["*.png", "*.bmp", "*.exr", "*.hdr", "*.jpg", "*.jpeg", "*.tga", "*.svg", "*.webp", ".ktx"]))
	open_file_dialog.set_file_mode(FileDialog.FILE_MODE_OPEN_FILE)
	open_file_dialog.access = FileDialog.ACCESS_FILESYSTEM

	dialog.add_child(save_file_dialog)
	dialog.add_child(open_file_dialog)


func _init_texture_picker(p_parent: Node, p_image_index: int) -> void:
	var line_edit: LineEdit = p_parent.find_child("LineEdit")
	var file_pick_button: Button = p_parent.find_child("PickButton")
	var clear_button: Button = p_parent.find_child("ClearButton")
	var texture_rect: TextureRect = p_parent.find_child("TextureRect")
	var texture_button: Button = p_parent.find_child("TextureButton")

	var open_fn: Callable = func() -> void:
		open_file_dialog.current_path = last_opened_directory
		if last_file_selected_fn != no_op:
			open_file_dialog.file_selected.disconnect(last_file_selected_fn)
		last_file_selected_fn = func(path: String) -> void: 
			line_edit.text = path
			line_edit.caret_column = path.length()
			last_opened_directory = path.get_base_dir() + "/"
			var image: Image = Image.new()
			var code: int = image.load(path)
			if code != OK:
				_show_error("Failed to load texture '" + path + "'")
				texture_rect.texture = null
				images[p_image_index] = null
			else:
				_show_success("Loaded texture '" + path + "'")
				texture_rect.texture = ImageTexture.create_from_image(image)
				images[p_image_index] = image
		open_file_dialog.file_selected.connect(last_file_selected_fn)
		open_file_dialog.popup_centered_ratio()

	var clear_fn: Callable = func() -> void:
		line_edit.text = ""
		texture_rect.texture = null
		images[p_image_index] = null

	# allow user to edit textbox and press enter because Godot's file picker doesn't work 100% of the time
	var line_edit_submit_fn: Callable = func(path: String) -> void: 
		var image: Image = Image.new()
		var code: int = image.load(path)
		if code != OK:
			_show_error("Failed to load texture '" + path + "'")
			texture_rect.texture = null
			images[p_image_index] = null
		else:
			texture_rect.texture = ImageTexture.create_from_image(image)
			images[p_image_index] = image	

	line_edit.text_submitted.connect(line_edit_submit_fn)
	file_pick_button.pressed.connect(open_fn)
	texture_button.pressed.connect(open_fn)
	clear_button.pressed.connect(clear_fn)
	_set_button_icon(file_pick_button, "Folder")
	_set_button_icon(clear_button, "Remove")


func _set_button_icon(p_button: Button, p_icon_name: String) -> void:
	var editor_base: Control = editor_interface.get_base_control()
	var icon: Texture2D = editor_base.get_theme_icon(p_icon_name, "EditorIcons")
	p_button.icon = icon


func _show_error(p_text: String) -> void:
	push_error("Terrain3DChannelPacker: " + p_text)
	status_label.text = p_text
	status_label.add_theme_color_override("font_color", Color(0.9, 0, 0))


func _show_success(p_text: String) -> void:
	print("Terrain3DChannelPacker: " + p_text)
	status_label.text = p_text
	status_label.add_theme_color_override("font_color", Color(0, 0.82, 0.14))


func _create_import_file(png_path: String) -> void:
	var dst_import_path: String = png_path + ".import"

	var file: FileAccess = FileAccess.open(TEMPLATE_PATH, FileAccess.READ)
	var template_content: String = file.get_as_text()
	file.close()

	var import_content: String = template_content.replace("$SOURCE_FILE", png_path)
	file = FileAccess.open(dst_import_path, FileAccess.WRITE)
	file.store_string(import_content)
	file.close()


func _on_pack_button_pressed() -> void:
	packing_albedo = images[IMAGE_ALBEDO] != null and images[IMAGE_HEIGHT] != null
	var packing_normal_roughness: bool = images[IMAGE_NORMAL] != null and images[IMAGE_ROUGHNESS] != null

	if not packing_albedo and not packing_normal_roughness:
		_show_error("Please select an albedo and height texture or a normal and roughness texture.")
		return

	if packing_albedo:
		save_file_dialog.current_path = last_saved_directory + "packed_albedo_height"
		save_file_dialog.title = "Save Packed Albedo/Height Texture"
		save_file_dialog.popup_centered_ratio()
		if packing_normal_roughness:
			queue_pack_normal_roughness = true
		return
	if packing_normal_roughness:
		save_file_dialog.current_path = last_saved_directory + "packed_normal_roughness"
		save_file_dialog.title = "Save Packed Normal/Roughness Texture"
		save_file_dialog.popup_centered_ratio()


func _on_save_file_selected(p_dst_path) -> void:
	last_saved_directory = p_dst_path.get_base_dir() + "/"
	if packing_albedo:
		_pack_textures(images[IMAGE_ALBEDO], images[IMAGE_HEIGHT], p_dst_path, false)
	else:
		_pack_textures(images[IMAGE_NORMAL], images[IMAGE_ROUGHNESS], p_dst_path, invert_green_checkbox.button_pressed)
	
	if queue_pack_normal_roughness:
		queue_pack_normal_roughness = false
		packing_albedo = false
		save_file_dialog.current_path = last_saved_directory + "packed_normal_roughness"
		save_file_dialog.title = "Save Packed Normal/Roughness Texture"
		save_file_dialog.call_deferred("popup_centered_ratio")


func _pack_textures(p_rgb_image: Image, p_a_image: Image, p_dst_path: String, p_invert_green: bool) -> void:
	if p_rgb_image and p_a_image:
		if p_rgb_image.get_size() != p_a_image.get_size():
			_show_error("Textures must be the same size.")
			return

		var output_image: Image = Terrain3DUtil.pack_image(p_rgb_image, p_a_image, p_invert_green)

		if not output_image:
			_show_error("Failed to pack textures.")
			return

		output_image.save_png(p_dst_path)
		editor_interface.get_resource_filesystem().scan_sources()
		_create_import_file(p_dst_path)
		_show_success("Packed to " + p_dst_path + ".")
	else:
		_show_error("Failed to load one or more textures.")
