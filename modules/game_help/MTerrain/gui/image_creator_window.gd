@tool
extends Window

@onready var image_name = find_child("image_name")
@onready var create_button = find_child("create_button")
@onready var error_label = find_child("error_label")
@onready var image_format = find_child("image_format")
@onready var data_compress_option = find_child("data_compress_option")
@onready var file_compress = find_child("file_compress")

var mterrain

func _on_close_requested():
	queue_free()

func _on_create_pressed():
	create_image()
	queue_free()

func _on_image_name_text_changed(new_text):
	var disable = new_text in mterrain.get_image_list()	
	create_button.disabled = disable
	error_label.visible = disable

func create_image():	
	var format:int = image_format.get_selected_id()
	var uniform_name:String = image_name.text	
	var file_compress = file_compress.selected
	var data_compress = data_compress_option.selected
	var data_dir = mterrain.dataDir
	var region_grid_size = (mterrain.terrain_size) / mterrain.region_size
	var image_width = (mterrain.region_size * mterrain.get_base_size() ) / mterrain.get_h_scale()	
	
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
			img.fill(Color.BLACK)
			mres.insert_data(img.get_data(),uniform_name,format,data_compress,file_compress)
			ResourceSaver.save(mres,path)			
