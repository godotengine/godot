@tool
extends Window

signal layer_created

@onready var no_terrain_label = find_child("no_terrain")
@onready var uniform_name_line = find_child("uniform_name_line")
@onready var format_option = find_child("uniform_name_2")
@onready var uniform_name_empty_error = find_child("no_terrain2")
@onready var def_color_picker = find_child("ColorPickerButton")
@onready var layer_types:OptionButton = find_child("layer_types")
@onready var file_compress_option = find_child("file_compress")
@onready var compress_option = find_child("data_compress_option")
@onready var remove_uniform_list = find_child("remove_uniform_list")

const config_file_name:=".save_config.ini"

var region_grid_size:Vector2i
var image_width:int=0
var data_dir:=""
var is_init = false

var active_terrain

func _ready():
	layer_types.item_selected.connect(func(id): $ScrollContainer/VBoxContainer/def_color.visible = id == 0)	
	
	find_child("close_button").pressed.connect(_on_close_requested)
	
func set_terrain(input:MTerrain):
	active_terrain = input
	if input.terrain_size.x % input.region_size !=0:
		printerr("Terrain size.x is not divisible by its region size")
		return
	if input.terrain_size.x % input.region_size !=0:
		printerr("Terrain size.y is not divisible by its region size")
		return
	region_grid_size.x = input.terrain_size.x/input.region_size
	region_grid_size.y = input.terrain_size.x/input.region_size
	image_width = ((input.region_size*input.get_base_size())/input.get_h_scale())
	data_dir = input.dataDir
	is_init = true
	no_terrain_label.visible = false
	## Setting image list
	var first_path = data_dir.path_join("x0_y0.res")
	if not ResourceLoader.exists(first_path):
		return
	var mres = ResourceLoader.load(first_path)
	if not (mres is MResource):
		return
	var data_names:Array= mres.compressed_data.keys()
	#for dname in data_names:
	#	remove_uniform_list.add_item(dname)
	#remove_uniform_list.select(-1)

func _on_close_requested():
	queue_free()
	

func _on_create_button_up():
	if not is_init:
		printerr("Image create/remove is not init")
		return
	var format:int = format_option.get_selected_id()
	var uniform_name:String = uniform_name_line.text
	var def_color:Color=def_color_picker.color
	var file_compress = file_compress_option.selected
	var compress = compress_option.selected
	
	if uniform_name.is_empty():
		printerr("Uniform Name is empty")
		uniform_name_empty_error.visible = true
		return
	else:
		uniform_name_empty_error.visible = false
	var dir = DirAccess.open(data_dir)
	if not dir:
		printerr("Can not open ",data_dir)
		return
	for j in range(region_grid_size.y):
		for i in range(region_grid_size.x):
			var path = data_dir.path_join("x"+str(i)+"_y"+str(j)+".res")
			var mres:MResource
			if ResourceLoader.exists(path):
				mres = ResourceLoader.load(path)
			else:
				mres = MResource.new()
			if not (mres is MResource):
				continue
			var img:Image = Image.create(image_width,image_width,false,format)
			img.fill(def_color)
			mres.insert_data(img.get_data(),uniform_name,format,compress,file_compress)
			ResourceSaver.save(mres,path)
			init_new_color_layer(uniform_name, def_color)
	#queue_free()

func init_new_color_layer(uniform_name, color):
	active_terrain.brush_layers_groups_num += 1
	var id = active_terrain.brush_layers_groups_num-1
	active_terrain.brush_layers[id] = MBrushLayers.new()
	active_terrain.brush_layers[id].layers_title = uniform_name
	active_terrain.brush_layers[id].uniform_name = uniform_name
	active_terrain.brush_layers[id].brush_name = layer_types.get_item_text(layer_types.selected)
	active_terrain.brush_layers[id].layers_num = 1
	active_terrain.brush_layers[id].layers[0].NAME = "background"
	if layer_types.selected == 0:
		active_terrain.brush_layers[id].layers[0].color = color
	layer_created.emit(active_terrain.brush_layers[id])
	if active_terrain.is_grid_created():
		active_terrain.restart_grid()
	else:
		active_terrain.create_grid()

func _on_remove_button_up():
	if not is_init:
		printerr("Image create/remove is not init")
		return
	var index = remove_uniform_list.selected
	if index < 0:
		return
	var dname = remove_uniform_list.get_item_text(index)
	var dir = DirAccess.open(data_dir)
	if not dir:
		printerr("Can not open ",data_dir)
		return
	dir.list_dir_begin()
	var file_name :String= dir.get_next()
	var res_names:PackedStringArray
	while file_name != "":
		if file_name.get_extension() == "res":
			res_names.append(file_name)
		file_name = dir.get_next()
	remove_config_file(dname)
	for res_name in res_names:
		var path = data_dir.path_join(res_name)
		var mres = load(path)
		if not (mres is MResource):
			continue
		mres.remove_data(dname)
		ResourceSaver.save(mres,path)
	#queue_free()


func update_config_file():
	var path = data_dir.path_join(config_file_name)
	var config = ConfigFile.new()
	if FileAccess.file_exists(path):
		var err = config.load(path)
		if err != OK:
			printerr("Can not load config with err ",err)
	
	var uniform_name:String = uniform_name_line.text
	var file_compress = file_compress_option.selected
	var compress = compress_option.selected
	
	config.set_value(uniform_name,"compress",compress)
	config.set_value(uniform_name,"file_compress",file_compress)
	config.save(path)


func remove_config_file(dname):
	var path = data_dir.path_join(config_file_name)
	var config = ConfigFile.new()
	if FileAccess.file_exists(path):
		var err = config.load(path)
		if err != OK:
			printerr("Can not load config with err ",err)
			return
	else:
		return
	
	config.erase_section(dname)
	config.save(path)
