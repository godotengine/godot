@tool
extends Terrain3D


@export var clear_all: bool = false : set = reset_settings
@export var clear_terrain: bool = false : set = reset_terrain
@export var update_height_range: bool = false : set = update_heights


func reset_settings(p_value) -> void:
	if p_value:
		height_file_name = ""
		control_file_name = ""
		color_file_name = ""
		import_position = Vector2i.ZERO
		height_offset = 0.0
		import_scale = 1.0
		r16_range = Vector2(0, 1)
		r16_size = Vector2i(1024, 1024)
		storage = null
		material = null
		assets = null


func reset_terrain(p_value) -> void:
	if p_value:
		storage = null


func update_heights(p_value) -> void:
	if p_value and storage:
		storage.update_height_range()


@export_group("Import File")
@export_global_file var height_file_name: String = ""
@export_global_file var control_file_name: String = ""
@export_global_file var color_file_name: String = ""
@export var import_position: Vector2i = Vector2i(0, 0) : set = set_import_position
@export var import_scale: float = 1.0
@export var height_offset: float = 0.0
@export var r16_range: Vector2 = Vector2(0, 1)
@export var r16_size: Vector2i = Vector2i(1024, 1024) : set = set_r16_size
@export var run_import: bool = false : set = start_import


func set_import_position(p_value: Vector2i) -> void:
	import_position.x = clamp(p_value.x, -8192, 8192)
	import_position.y = clamp(p_value.y, -8192, 8192)


func set_r16_size(p_value: Vector2i) -> void:
	r16_size.x = clamp(p_value.x, 0, 16384)
	r16_size.y = clamp(p_value.y, 0, 16384)


func start_import(p_value: bool) -> void:
	if p_value:
		print("Terrain3DImporter: Importing files:\n\t%s\n\t%s\n\t%s" % [ height_file_name, control_file_name, color_file_name])
		if not storage:
			storage = Terrain3DStorage.new()

		var imported_images: Array[Image]
		imported_images.resize(Terrain3DStorage.TYPE_MAX)
		var min_max := Vector2(0, 1)
		var img: Image
		if height_file_name:
			img = Terrain3DUtil.load_image(height_file_name, ResourceLoader.CACHE_MODE_IGNORE, r16_range, r16_size)
			min_max = Terrain3DUtil.get_min_max(img)
			imported_images[Terrain3DStorage.TYPE_HEIGHT] = img
		if control_file_name:
			img = Terrain3DUtil.load_image(control_file_name, ResourceLoader.CACHE_MODE_IGNORE)
			imported_images[Terrain3DStorage.TYPE_CONTROL] = img
		if color_file_name:
			img = Terrain3DUtil.load_image(color_file_name, ResourceLoader.CACHE_MODE_IGNORE)
			imported_images[Terrain3DStorage.TYPE_COLOR] = img
			if assets.get_texture_count() == 0:
				material.show_checkered = false
				material.show_colormap = true
		var pos := Vector3(import_position.x, 0, import_position.y)
		storage.import_images(imported_images, pos, height_offset, import_scale)
		print("Terrain3DImporter: Import finished")


@export_group("Export File")
enum { TYPE_HEIGHT, TYPE_CONTROL, TYPE_COLOR }
@export_enum("Height:0", "Control:1", "Color:2") var map_type: int = TYPE_HEIGHT
@export var file_name_out: String = ""
@export var run_export: bool = false : set = start_export

func start_export(p_value: bool) -> void:
	var err: int = storage.export_image(file_name_out, map_type)
	print("Terrain3DImporter: Export error status: ", err, " ", error_string(err))
	
