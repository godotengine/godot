@tool
extends Window

@onready var err := $tab/import/err

@onready var fileDialog = $FileDialog
@onready var fileDialog_save_folder = $FileDialog_save
@onready var region_container = $tab/import/HBoxContainer2
@onready var save_folder_line = $tab/import/save/save_folder_line
@onready var select_file_line: = $tab/import/HBoxContainer/filepath_line
@onready var image_dimension_root: = $tab/import/image_dimension
@onready var image_width_line: = $tab/import/image_dimension/width
@onready var image_height_line: = $tab/import/image_dimension/height
@onready var min_height_root:= $tab/import/min_height
@onready var max_height_root:= $tab/import/max_height
@onready var unform_name_line:=$tab/import/uniform_name/uniform_name_line
@onready var is_heightmap_checkbox:= $tab/import/is_heightmap_checkbox
@onready var min_height_line := $tab/import/min_height/min_height_line
@onready var max_height_line := $tab/import/max_height/max_height_line
@onready var region_size_line:= $tab/import/HBoxContainer2/region_size_line
@onready var width_line:= $tab/import/image_dimension/width
@onready var height_line:= $tab/import/image_dimension/height
@onready var image_format_option:=$tab/import/uniform_name2/image_format_option
@onready var flips_container := $tab/import/flips
@onready var flip_x_checkbox := $tab/import/flips/flip_x
@onready var flip_y_checkbox := $tab/import/flips/flip_y

@onready var accuracy_container := $tab/import/uniform_name3
@onready var accuracy_line := $tab/import/uniform_name3/accuracy
@onready var compress_qtq_checkbox := $tab/import/compress_qtq
@onready var data_compress_option := $tab/import/data_compress_option
@onready var file_compress_option := $tab/import/file_compress

const format_RF_index = 8
const config_file_name := ".save_config.ini"

var file_path:String
var ext:String
var save_path:String
var tmp_path:String
var region_size:int
var unifrom_name:String
var width:int
var height:int
var min_height:float
var max_height:float
var image_format:int
var flip_x:bool
var flip_y:bool
var accuracy:float
var compress_qtq:bool
var file_compress:int
var compress:int

var is_heightmap:=false

func _on_close_requested():
	queue_free()

func _on_button_button_down():
	fileDialog.visible = true

func _on_file_dialog_files_selected(paths):
	var path:String = paths[0]
	select_file_line.text = path 
	_on_filepath_line_text_changed(path)

func _on_filepath_line_text_changed(new_text:String):
	var ext = new_text.get_extension()
	image_dimension_root.visible = ext == "r16"
	var x = get_integer_inside_string("x",new_text)
	var y = get_integer_inside_string("y",new_text)
	var is_tiled = (not x==-1 and not y==-1)
	flips_container.visible = is_tiled
	region_container.visible = not is_tiled

func _on_check_button_toggled(button_pressed):
	min_height_root.visible = button_pressed
	max_height_root.visible = button_pressed
	accuracy_container.visible = button_pressed
	compress_qtq_checkbox.visible = button_pressed
	unform_name_line.editable = not button_pressed
	data_compress_option.visible = not button_pressed
	is_heightmap = button_pressed
	if(button_pressed):
		unform_name_line.text = "heightmap"
		image_format_option.select(format_RF_index)
		image_format_option.disabled = true
		
	elif unform_name_line.text == "heightmap":
		unform_name_line.text = ""
		image_format_option.disabled = false

func _on_save_folder_button_pressed():
	fileDialog_save_folder.visible = true

func _on_file_dialog_save_dir_selected(dir):
	save_folder_line.text = dir

func get_integer_inside_string(prefix:String,path:String)->int:
	path = path.to_lower()
	var reg = RegEx.new()
	reg.compile(prefix+"(\\d+)")
	var result := reg.search(path)
	if not result:
		return -1
	return result.strings[1].to_int()

func _on_import_pressed():
	err.visible = false
	file_path= select_file_line.text
	ext = select_file_line.text.get_extension()
	save_path= save_folder_line.text
	region_size = region_size_line.text.to_int()
	unifrom_name = unform_name_line.text
	width = width_line.text.to_int()
	height = height_line.text.to_int()
	min_height = min_height_line.text.to_float()
	max_height = max_height_line.text.to_float()
	image_format = image_format_option.get_selected_id()
	flip_x = flip_x_checkbox.button_pressed
	flip_y = flip_y_checkbox.button_pressed
	compress_qtq = compress_qtq_checkbox.button_pressed
	file_compress = file_compress_option.selected
	compress = data_compress_option.selected
	accuracy = float(accuracy_line.text)
	if is_heightmap and accuracy < 0.000001:
		perr("Accuracy can not be less than 0.000001")
		return
	if unifrom_name == "":
		perr("Uniform name is empty")
		return
	var x = get_integer_inside_string("x",file_path)
	var y = get_integer_inside_string("y",file_path)
	#In this case there is no tile and we should tile that
	if(x==-1 or y==-1):
		import_no_tile()
	else: #And in this case there is tiled already and regions size will ignored
		import_tile()

func is_valid_2n_plus_one(input:int):
	if input<3:
		return false
	input -= 1
	while true:
		if input == 1:
			return true
		if input%2!=0:
			return false
		input /=2

func is_power_of_two(input:int):
	while true:
		if input == 1:
			return true
		if input%2!=0:
			return false
		input /=2

func import_no_tile():
	if(region_size<32):
		perr("Region size can not be smaller than 32")
		return
	if(not is_power_of_two(region_size)):
		perr("Region size must be 2^n, like 16, 32, 256 ...")
		return
	if(save_path.is_empty()):
		perr("Save path is empty")
		return
	var img:Image
	if ext=="r16":
		img=MTool.get_r16_image(file_path,width,height,min_height,max_height,false)
	else:
		img = Image.load_from_file(file_path)
	if not img:
		perr("Can not load image")
		return
	var img_size = img.get_size()
	if ext!="r16" and is_heightmap:
		img.convert(Image.FORMAT_RF)
		var data = img.get_data().to_float32_array()
		for i in range(0,data.size()):
			data[i] *= (max_height - min_height)
			data[i] += min_height
		img = Image.create_from_data(img_size.x,img_size.y,false,Image.FORMAT_RF, data.to_byte_array())
	var region_grid_size:= Vector2i()
	
	region_grid_size.x = ceil(float(img_size.x)/(region_size))
	region_grid_size.y = ceil(float(img_size.y)/(region_size))
	var total_regions = region_grid_size.x*region_grid_size.y
	if(total_regions>9000000):
		perr("make region size bigger, too many regions, region count: "+str(total_regions))
		return
	if is_heightmap:
		update_config_file_for_heightmap()
	else:
		update_config_file_for_data()
	for y in range(0, region_grid_size.x):
		for x in range(0, region_grid_size.y):
			var r_save_name:String = "x"+str(x)+"_y"+str(y)+".res"
			var r_path:String = save_path.path_join(r_save_name)
			var pos:=Vector2i(x,y)
			pos *= (region_size)
			if abs(pos.x - img_size.x) < 2 or abs(pos.y - img_size.y) < 2:
				continue
			var rect = Rect2i(pos, Vector2i(region_size,region_size));
			var region_img:= img.get_region(rect)
			if(image_format>=0 and not is_heightmap):
				region_img.convert(image_format)
			var mres:MResource
			if ResourceLoader.exists(r_path):
				var mres_loaded = ResourceLoader.load(r_path)
				if mres_loaded:
					if mres_loaded is MResource:
						mres = mres_loaded
			if not mres:
				mres = MResource.new()
			if is_heightmap:
				mres.insert_heightmap_rf(region_img.get_data(),accuracy,compress_qtq,file_compress)
			else:
				mres.insert_data(region_img.get_data(),unifrom_name,region_img.get_format(),compress,file_compress)
			ResourceSaver.save(mres,r_path)
	queue_free()

func get_file_path(x:int,y:int,path_pattern:String)->String:
	var regx = RegEx.new()
	var regy = RegEx.new()
	var patternx = "(?i)(x)(\\d+)"
	var paterrny = "(?i)(y)(\\d+)"
	regx.compile(patternx)
	regy.compile(paterrny)
	var resx = regx.search(path_pattern)
	var resy = regy.search(path_pattern)
	## This is because in some cases we have x1 in other cases we have x01
	for i in range(1,4):
		var digit_pattern = "%0"+str(i)+"d"
		var xstr = digit_pattern % x
		var ystr = digit_pattern % y
		var sub = regx.sub(path_pattern, resx.strings[1]+xstr)
		sub = regy.sub(sub, resy.strings[1]+ystr)
		if FileAccess.file_exists(sub):
			return sub
	return ""

func import_tile():
	var tiled_files:Dictionary
	var x:int=0
	var y:int=0
	var x_size = 0
	var y_size = 0
	while true:
		var r_path = get_file_path(x,y,file_path)
		if(r_path == ""):
			x = 0
			y += 1
			r_path = get_file_path(x,y,file_path)
			if(r_path == ""):
				if tiled_files.is_empty():
					perr("Can't find first tile x=0 and y=0")
					return
				break
		tiled_files[Vector2i(x,y)] = r_path
		if x > x_size: x_size = x
		if y > y_size: y_size = y
		x+=1
	if flip_y:
		var tmp:Dictionary
		var j = y_size
		var oj = 0
		while  j>= 0:
			for i in range(0,x_size+1):
				tmp[Vector2i(i,oj)] = tiled_files[Vector2i(i,j)]
			j-=1
			oj+=1
		tiled_files = tmp
	if flip_x:
		var tmp:Dictionary
		var i = y_size
		var oi = 0
		while  i>= 0:
			for j in range(0,y_size+1):
				tmp[Vector2i(oi,j)] = tiled_files[Vector2i(i,j)]
			i-=1
			oi+=1
		tiled_files = tmp
	for i in range(0,x_size+1):
		for j in range(0,y_size+1):
			var r_path = tiled_files[Vector2i(i,j)]
			var r_save_name:String = "x"+str(i)+"_y"+str(j)+".res"
			var r_save_path:String = save_path.path_join(r_save_name)
			var img:Image
			if ext == "r16":
				img=MTool.get_r16_image(r_path,0,0,min_height,max_height,false)
			else:
				img = Image.load_from_file(r_path)
			if not img:
				perr("Can not load image")
				return
			if img.get_size().x != img.get_size().y:
				perr("In tiled mode image width and height should be equal")
				return
			if not is_power_of_two(img.get_size().x):
				perr("In tiled mode image height and width should be in power of two")
				return
			var img_size = img.get_size().x
			if ext!="r16" and is_heightmap:
				img.convert(Image.FORMAT_RF)
				var data = img.get_data().to_float32_array()
				for d in range(0,data.size()):
					data[d] *= (max_height - min_height)
					data[d] += min_height
				img = Image.create_from_data(img_size,img_size,false,Image.FORMAT_RF, data.to_byte_array())
			if(image_format>=0 and not is_heightmap):
				img.convert(image_format)
			print("path ", img.get_path())
			var mres:MResource
			if ResourceLoader.exists(r_save_path):
				var mres_loaded = ResourceLoader.load(r_save_path)
				if mres_loaded:
					if mres_loaded is MResource:
						mres = mres_loaded
			if not mres:
				mres = MResource.new()
			if is_heightmap:
				mres.insert_heightmap_rf(img.get_data(),accuracy,compress_qtq,file_compress)
			else:
				mres.insert_data(img.get_data(),unifrom_name,img.get_format(),compress,file_compress)
			ResourceSaver.save(mres, r_save_path)
	### Now Correcting the edges
	#correct_edges(unifrom_name, tmp_path)
	queue_free()


func get_img_or_black(x:int,y:int,u_name:String,dir:String,size:Vector2i,format:int)->Image:
	var file_name = u_name + "_x"+str(x)+"_y"+str(y)+".res"
	var path = dir.path_join(file_name)
	if ResourceLoader.exists(path):
		return load(path)
	else:
		var img:= Image.create(size.x,size.y,false,format)
		img.fill(Color(-10000000000, 0,0))
		return img

func correct_edges(u_name:String, dir:String):
	var x:int =0
	var y:int =0
	while true:
		var file_name = u_name + "_x"+str(x)+"_y"+str(y)+".res"
		var path = dir.path_join(file_name)
		if !ResourceLoader.exists(path):
			x = 0
			y+=1
			file_name = u_name + "_x"+str(x)+"_y"+str(y)+".res"
			path = dir.path_join(file_name)
			if !ResourceLoader.exists(path):
				break
		var img:Image = load(path)
		var size = img.get_size()
		var right_img:Image = get_img_or_black(x+1,y,u_name,dir,size,img.get_format())
		var bottom_img:Image = get_img_or_black(x,y+1,u_name,dir,size,img.get_format())
		var right_bottom_img:Image = get_img_or_black(x+1,y+1,u_name,dir,size,img.get_format())
		img = img.get_region(Rect2i(0,0,size.x+1,size.y+1))
		##Correct right side
		for j in range(0,size.y):
			var col = right_img.get_pixel(0, j)
			img.set_pixel(size.x, j, col)
		##Correct bottom side
		for i in range(0,size.x):
			var col = bottom_img.get_pixel(i, 0)
			img.set_pixel(i , size.y, col)
		##Correct right bottom corner
		var col = right_bottom_img.get_pixel(0,0)
		img.set_pixel(size.x , size.y, col)
		var save_name = u_name + "_x"+str(x)+"_y"+str(y)+".res"
		var r_save_path = save_path.path_join(save_name)
		ResourceSaver.save(img, r_save_path)
		print("save ", r_save_path)
		x+=1


func perr(msg:String):
	err.visible = true
	err.text = msg
	printerr(msg)


func update_config_file_for_heightmap():
	var path = save_path.path_join(config_file_name)
	var conf := ConfigFile.new()
	if FileAccess.file_exists(path):
		var err = conf.load(path)
		if err != OK:
			printerr("Can not load conf file")
	conf.set_value("heightmap","accuracy",accuracy)
	conf.set_value("heightmap","file_compress",file_compress)
	conf.set_value("heightmap","compress_qtq",compress_qtq)
	conf.save(path)

func update_config_file_for_data():
	var path = save_path.path_join(config_file_name)
	var conf := ConfigFile.new()
	if FileAccess.file_exists(path):
		var err = conf.load(path)
		if err != OK:
			printerr("Can not load conf file")
	conf.set_value(unifrom_name,"compress",compress)
	conf.set_value(unifrom_name,"file_compress",file_compress)
	conf.save(path)


###################### Export
@onready var export_err := $tab/export/export_err
@onready var einfo := $tab/export/einfo
@onready var edata_name_option := $tab/export/edata_name_option
@onready var eformat_option := $tab/export/eformat_option
@onready var export_hc := $tab/export/export_hc
@onready var emin_line := $tab/export/export_hc/min
@onready var emax_line := $tab/export/export_hc/max
@onready var export_path_line := $tab/export/HBoxContainer/LineEdit
@onready var file_dialog_export := $FileDialog_export

var active_terrain:MTerrain
const format_for_heightmap := ["OpenEXR","raw16","Godot res"]
const format_for_data := ["PNG","Godot res"]
const format_for_all := ["Godot res"]

var ednames:PackedStringArray

func init_export(_t:MTerrain):
	edata_name_option.clear()
	eformat_option.clear()
	export_err.visible = false
	einfo.visible = false
	var first_path = _t.dataDir.path_join("x0_y0.res")
	if not ResourceLoader.exists(first_path):
		print_export_err("Can not find Data "+first_path)
		return
	var mres:MResource
	var mres_load = ResourceLoader.load(first_path)
	if not mres_load:
		print_export_err("Can not load "+first_path)
		return
	if not(mres_load is MResource):
		print_export_err(first_path+" is not a MResource type")
		return
	mres = mres_load
	var data_names = mres.compressed_data.keys()
	for dname in data_names:
		edata_name_option.add_item(dname)
		ednames.push_back(dname)
	edata_name_option.add_item("all")
	edata_name_option.select(-1)
	active_terrain = _t 



func _on_edata_name_option_item_selected(index):
	if index < 0: return
	var dname = edata_name_option.get_item_text(index)
	eformat_option.clear()
	var is_h = dname == "heightmap"
	export_hc.visible = is_h
	einfo.visible = true
	if is_h:
		einfo.text="Min and Max height are used for normalizing heightmap before export, Godot res format does need this, Auto detect can take a while"
	elif dname != "all":
		einfo.text="PNG only support: L8,RGB8,RGBA8 using other format will automatically converted to these"
	else:
		einfo.visible = false
	if is_h:
		for o in format_for_heightmap:
			eformat_option.add_item(o)
	elif dname == "all":
		einfo.visible
		for o in format_for_all:
			eformat_option.add_item(o)
	else:
		for o in format_for_data:
			eformat_option.add_item(o)
	eformat_option.select(-1)

func _on_auto_min_max_detect_button_up():
	if not active_terrain:
		print_export_err("No active Terrain")
		return
	var first_path = active_terrain.dataDir.path_join("x0_y0.res")
	if not ResourceLoader.exists(first_path):
		print_export_err("Can not find Data "+first_path)
		return
	var mres:MResource
	var mres_load = ResourceLoader.load(first_path)
	if not mres_load:
		print_export_err("Can not load "+first_path)
		return
	if not(mres_load is MResource):
		print_export_err(first_path+" is not a MResource type")
		return
	mres = mres_load
	var min_height:float=mres.get_min_height()
	var max_height:float=mres.get_max_height()
	var x:int=1
	var y:int=0
	while true:
		var path = active_terrain.dataDir.path_join("x"+str(x)+"_y"+str(y)+".res")
		if not ResourceLoader.exists(path):
			y+=1
			x=0
			path = active_terrain.dataDir.path_join("x"+str(x)+"_y"+str(y)+".res")
			if not ResourceLoader.exists(path):
				break
		x+=1
		mres_load = ResourceLoader.load(path)
		if not mres_load:
			continue
		if not (mres_load is MResource):
			continue
		mres = mres_load
		if min_height > mres.get_min_height():
			min_height = mres.get_min_height()
		if max_height < mres.get_max_height():
			max_height = mres.get_max_height()
	emin_line.text = str(min_height)
	emax_line.text = str(max_height)

func _on_export_path_btn_button_up():
	file_dialog_export.visible = true

func _on_file_dialog_export_dir_selected(dir):
	export_path_line.text = dir

func _on_export_btn_button_up():
	if edata_name_option.selected < 0:
		print_export_err("No data name selected")
		return
	var selected_dname = edata_name_option.get_item_text(edata_name_option.selected)
	if selected_dname.is_empty():
		print_export_err("data name is empty")
		return
	if eformat_option.selected < 0:
		print_export_err("No export format selected")
		return
	var export_format = eformat_option.get_item_text(eformat_option.selected)
	if export_format.is_empty():
		print_export_err("Format name is empty")
		return
	if not active_terrain:
		print_export_err("No active Terrain")
		return
	if selected_dname == "all":
		for ee in ednames:
			export_data(ee,export_format)
	else:
		export_data(selected_dname,export_format)
	queue_free()

func export_data(dname:String,eformat:String):
	var export_path :String = export_path_line.text
	var min_height:float = float(emin_line.text)
	var max_height:float = float(emax_line.text)
	if (max_height - min_height <=0.000001) and dname == "heightmap" and eformat != "Godot res":
		print_export_err("Min and Max height are not set")
		return
	if export_path.is_empty():
		print_export_err("Export path is empty")
		return
	if not export_path.is_absolute_path():
		print_export_err("Export path is not a valid absoulute path")
		return
	var mres_load
	var mres:MResource
	var x:int=0
	var y:int=0
	while true:
		var path = active_terrain.dataDir.path_join("x"+str(x)+"_y"+str(y)+".res")
		if not ResourceLoader.exists(path):
			y+=1
			x=0
			path = active_terrain.dataDir.path_join("x"+str(x)+"_y"+str(y)+".res")
			if not ResourceLoader.exists(path):
				break
		mres_load = ResourceLoader.load(path)
		if not mres_load:
			print_export_err("Can not load "+path)
			continue
		if not (mres_load is MResource):
			print_export_err(path+" is not MResource type")
			continue
		mres = mres_load
		var data:PackedByteArray
		var img_format:int=-1
		var width:int
		if dname == "heightmap":
			data = mres.get_heightmap_rf(false)
			img_format = Image.FORMAT_RF
		else:
			data = mres.get_data(dname,false)
		if data.size() == 0:
			print_export_err("Something wrong can not save "+path)
			continue
		width = mres.get_data_width(dname)
		if dname != "heightmap":
			img_format = mres.get_data_format(dname)
		var path_no_ext = export_path.path_join(dname+"_x"+str(x)+"_y"+str(y))
		if eformat == "raw16":
			var final_epath = path_no_ext + ".r16"
			MTool.write_r16(final_epath,data,min_height,max_height)
		elif eformat == "OpenEXR":
			var final_epath = path_no_ext + ".exr"
			if dname == "heightmap":
				data = MTool.normalize_rf_data(data,min_height,max_height)
			var img:=Image.create_from_data(width,width,false,img_format,data)
			if dname == "heightmap":
				img.save_exr(final_epath,true)
			else:
				img.save_exr(final_epath)
		elif eformat == "PNG":
			var final_epath = path_no_ext + ".png"
			var img:=Image.create_from_data(width,width,false,img_format,data)
			img.save_png(final_epath)
		elif eformat == "Godot res":
			var final_epath = path_no_ext + ".res"
			var img:=Image.create_from_data(width,width,false,img_format,data)
			ResourceSaver.save(img,final_epath)
		else:
			print_export_err("Unknow export format")
			return
		x+=1


func print_export_err(msg:String):
	export_err.visible = true
	export_err.text = msg
	printerr(msg)
