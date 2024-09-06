@tool
extends Window

signal layer_created

@onready var layer_name_input:LineEdit = find_child("layer_name")
@onready var uniform_name_input:LineEdit = find_child("uniform_name")
@onready var format_option:OptionButton = find_child("image_format")
@onready var def_color_picker = find_child("ColorPickerButton")
@onready var layer_types:OptionButton = find_child("layer_types")
@onready var file_compress_option = find_child("file_compress")
@onready var compress_option: OptionButton = find_child("data_compress_option")
@onready var instructions_label: Label = find_child("instructions_label")
@onready var create_button:Button = find_child("create")

@onready var advanced_settings_button:Button = find_child("advanced_settings_button")
@onready var advanced_settings_control:Control = find_child("advanced_settings")


const config_file_name:=".save_config.ini"

var region_grid_size:Vector2i
var image_width:int=0
var data_dir:=""
var is_init = false

var active_terrain: MTerrain

var advanced_settings = false
var uniform_exists = false

func _ready():
	layer_name_input.text_changed.connect(func(new_text):
		validate_settings()
	)
	uniform_name_input.text_changed.connect(func(new_text):
		validate_settings()
	)
	layer_types.item_selected.connect(func(id): 
		find_child("def_color").visible = id == 0 #color paint
		var allowed_formats = []		
		format_option.select(Image.FORMAT_RGBA8)
		if layer_types.get_item_text(id) == "Paint 16":
			format_option.select(Image.FORMAT_RGBA4444)			
			allowed_formats = [Image.FORMAT_RGBA4444]
		if layer_types.get_item_text(id) == "Paint 256":
			format_option.select(Image.FORMAT_R8)			
			allowed_formats = [Image.FORMAT_R8, Image.FORMAT_RG8,Image.FORMAT_RGB8,Image.FORMAT_RGBA8]
		
		for i in format_option.item_count:
			if allowed_formats == []:
				format_option.set_item_disabled(i, false)	
			else:
				format_option.set_item_disabled(i, not i in allowed_formats)	
		
		validate_settings()		
	)	
	format_option.item_selected.connect(func(id):		
		find_child("data_compress_settings").visible = id in [Image.FORMAT_L8, Image.FORMAT_RGB8, Image.FORMAT_RGBA8]
		compress_option.set_item_disabled(1, not id in [Image.FORMAT_RGB8, Image.FORMAT_RGBA8])
		compress_option.set_item_disabled(2, not id in [Image.FORMAT_L8, Image.FORMAT_RGB8, Image.FORMAT_RGBA8])		
		validate_settings()
	)
	find_child("close_button").pressed.connect(_on_close_requested)
	create_button.button_up.connect(_on_create_button_up)

	advanced_settings_button.pressed.connect(func():
		advanced_settings_button.visible = false
		advanced_settings_control.visible = true
		advanced_settings = true		
	)		
	
func validate_settings():
	var instructions = ""
	var warnings = ""
	var existing_layers = active_terrain.brush_layers.map(func(a):return a.layers_title)
	advanced_settings_button.visible = true	
	advanced_settings_control.visible = false
	if layer_name_input.text.strip_edges() == "":
		instructions += "Please enter a Layer Name\n"		
	elif layer_name_input.text == "" or layer_name_input.text in existing_layers:
		instructions += "Layer name aleady exists.\n"
	if uniform_name_input.text == "":
		instructions += "Please enter a Uniform Name\n"
	elif uniform_name_input.text in active_terrain.brush_layers.map(func(a): return a.uniform_name):		
		var others_mode = active_terrain.brush_layers.filter(func(a): return a.uniform_name == uniform_name_input.text).map(func(a): return a.brush_name)		
		if "Color Paint" in others_mode or layer_types.selected == 0:
			instructions += "Uniform already exists. Cannot share uniform in Color Paint mode"		
		warnings += "This layer is about to share a Uniform with another layer. Please be careful with how you combine brushes"
		advanced_settings_button.visible = false	
		advanced_settings_control.visible = false		
		uniform_exists = true
	elif uniform_name_input.text in active_terrain.get_image_list():
		warnings += "This layer is about to connect with an existing uniform image"		
		uniform_exists = true
	else:
		uniform_exists = false
		warnings += str("Creating new image uniform. \nTo use in shader you need to add \n`uniform sampler2d mterrain_",uniform_name_input.text, "` to your shader \nthen restart the terrain")
	
	instructions_label.text = warnings if instructions == "" else instructions 	
	create_button.disabled = instructions != ""
	
	
func set_terrain(input:MTerrain):
	active_terrain = input
	active_terrain.save_all_dirty_images()
	validate_settings.call_deferred()
	if input.terrain_quad_count.x % input.region_quad_count !=0:
		printerr("Terrain size.x is not divisible by its region size")
		return
	if input.terrain_quad_count.x % input.region_quad_count !=0:
		printerr("Terrain size.y is not divisible by its region size")
		return
	region_grid_size.x = input.terrain_quad_count.x/input.region_quad_count
	region_grid_size.y = input.terrain_quad_count.x/input.region_quad_count
	image_width = ((input.region_quad_count*input.get_base_size())/input.get_h_scale())
	data_dir = input.dataDir
	is_init = true	
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
	var uniform_name:String = uniform_name_input.text
	var def_color:Color=def_color_picker.color
	var file_compress = file_compress_option.selected
	var compress = compress_option.selected	
	if not uniform_exists:
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
	else:
		pass
		#print("uniform exists")
	
	init_new_color_layer(layer_name_input.text, uniform_name, def_color)	
	queue_free()
		

func init_new_color_layer(layer_name, uniform_name, color = null):
	active_terrain.brush_layers_groups_num += 1
	var id = active_terrain.brush_layers_groups_num-1
	active_terrain.brush_layers[id] = MBrushLayers.new()
	active_terrain.brush_layers[id].layers_title = layer_name
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
