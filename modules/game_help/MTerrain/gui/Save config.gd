@tool
extends TabBar

@onready var main_window:=$"../../.."
@onready var error = $scroll/VBoxContainer/Error

@onready var compress_qtq:=$scroll/VBoxContainer/HBoxContainer/compress_qtq
@onready var accuracy:=$scroll/VBoxContainer/HBoxContainer2/LineEdit
@onready var hfile_compress:=$scroll/VBoxContainer/hfilecomrpess/file_compress
@onready var data_compress:=$scroll/VBoxContainer/hb2/data_compress_option
@onready var dfile_compress:=$scroll/VBoxContainer/hb3/file_compress
@onready var data_name_option:=$scroll/VBoxContainer/hb/data_name_option
@onready var is_force_apply_checkbox:=$scroll/VBoxContainer/force_apply_all


const default_accuracy:=0.02
const config_file_name:=".save_config.ini"

var is_init = false

var data_state_init:Dictionary
var data_state_final:Dictionary

var terrain:MTerrain

func init_save_config(_t:MTerrain):
	terrain = _t
	if not _t:
		error.text = "No Active terrain"
		return
	if _t.is_grid_created():
		error.text = "Destroy Terrain to be able apply changes"
		return
	
	data_name_option.clear()
	var data_dir = _t.dataDir
	var first_region_path = data_dir.path_join("x0_y0.res")
	if( not ResourceLoader.exists(first_region_path)):
		printerr("Can not find "+first_region_path)
		error.text = "Can not find "+first_region_path
		return
	var mres:MResource
	mres = ResourceLoader.load(first_region_path)
	if not mres:
		printerr("Can not load "+first_region_path)
		error.text = "Can not load "+first_region_path
		return
	error.text = ""
	error.visible = false
	var data_names = mres.compressed_data.keys()
	var config_file:=ConfigFile.new()
	var config_file_path = data_dir.path_join(config_file_name)
	if FileAccess.file_exists(config_file_path):
		var err = config_file.load(config_file_path)
		if err != OK:
			printerr("Can not Load config file with err ",err)
	var first_data_name:String=""
	for dname in data_names:
		var state:Dictionary
		var res_file_compress = mres.get_file_compress(dname)
		var format = mres.get_data_format(dname)
		state["format"] = format
		if dname == &"heightmap":
			if config_file.has_section_key("heightmap","accuracy"):
				state["accuracy"]=float(config_file.get_value("heightmap","accuracy"))
			else:
				state["accuracy"]=default_accuracy
			if config_file.has_section_key("heightmap","compress_qtq"):
				state["compress_qtq"]=bool(config_file.get_value("heightmap","compress_qtq"))
			else:
				state["compress_qtq"]= mres.is_compress_qtq()
			if config_file.has_section_key("heightmap","file_compress"):
				state["file_compress"]=int(config_file.get_value("heightmap","file_compress"))
			else:
				state["file_compress"]=res_file_compress
		else:
			var res_compress = mres.get_compress(dname)
			if config_file.has_section_key(dname,"file_compress"):
				state["file_compress"]=int(config_file.get_value(dname,"file_compress"))
			else:
				state["file_compress"] = res_file_compress
			if config_file.has_section_key(dname,"compress"):
				state["compress"]=int(config_file.get_value(dname,"compress"))
			else:
				state["compress"] = res_compress
		if dname == &"heightmap":
			compress_qtq.button_pressed = state["compress_qtq"]
			accuracy.text = str(state["accuracy"])
			hfile_compress.select(state["file_compress"])
		else:
			data_name_option.add_item(dname)
			if first_data_name.is_empty():
				first_data_name = dname
				dfile_compress.select(state["file_compress"])
				data_compress.select(state["compress"])
		data_state_init[dname]=state
	if not first_data_name.is_empty():
		data_name_option.select(0)
	data_state_final = data_state_init.duplicate(true)
	is_init = true

func _on_data_name_option_item_selected(index):
	var dname = data_name_option.get_item_text(index)
	if(not data_state_final.has(dname)):
		printerr("Can not find ",dname)
		return
	var state:Dictionary = data_state_final[dname]
	var keys := state.keys()
	for key in keys:
		if key == "file_compress":
			dfile_compress.select(state["file_compress"])
		elif key == "compress":
			data_compress.select(state["compress"])


func _on_data_compress_option_item_selected(index):
	var dname = data_name_option.get_item_text(data_name_option.selected)
	if(not data_state_final.has(dname)):
		printerr("Can not find ",dname)
		return
	var state:Dictionary = data_state_final[dname]
	state["compress"] = index
	data_state_final[dname] = state


func _on_file_compress_item_selected(index):
	var dname = data_name_option.get_item_text(data_name_option.selected)
	if(not data_state_final.has(dname)):
		printerr("Can not find ",dname)
		return
	var state:Dictionary = data_state_final[dname]
	state["file_compress"] = index
	data_state_final[dname] = state


func is_changed_image_state(dname:StringName):
	if dname!=&"heightmap":
		return data_state_final[dname]!=data_state_init[dname]
	var state_init = data_state_init[&"heightmap"]
	if state_init["compress_qtq"] != compress_qtq.button_pressed:
		return true
	if abs(state_init["accuracy"] - float(accuracy.text)) > 0.0001:
		return true
	if state_init["file_compress"] != hfile_compress.selected:
		return true
	return false


func _on_apply_button_up():
	var ac = float(accuracy.text)
	if ac < 0.0000001:
		printerr("Accuracy can not be less than 0.0000001")
		return
	if not terrain:
		printerr("No Valid Terrain")
		return
	if not is_init:
		printerr("No init")
		return
	if terrain.is_grid_created():
		printerr("destroy Terrain first")
		return
	var keys = data_state_init.keys()
	var is_force_apply = is_force_apply_checkbox.button_pressed
	var has_change = false
	for key in keys:
		if is_force_apply or is_changed_image_state(key):
			has_change = true
			break
	if not has_change:
		print("Nothing to change")
		return
	var dir = DirAccess.open(terrain.dataDir)
	if not dir:
		printerr("Can not open ",terrain.dataDir)
		return
	dir.list_dir_begin()
	var file_name :String= dir.get_next()
	var res_names:PackedStringArray
	while file_name != "":
		if file_name.get_extension() == "res":
			res_names.append(file_name)
		file_name = dir.get_next()
	
	save_config_file()
	for res_name in res_names:
		var res_path = terrain.dataDir.path_join(res_name)
		var mres = load(res_path)
		if not (mres is MResource):
			print("Not MResource")
			continue
		for key in keys:
			if not mres.compressed_data.has(key):
				printerr("Resource "+res_path+" does not have data "+key)
				continue
			if key == &"heightmap":
				if is_force_apply or is_changed_image_state(&"heightmap"):
					apply_update_heightmap(mres)
			elif is_force_apply or is_changed_image_state(key):
				apply_update_data(key,mres)
		ResourceSaver.save(mres,res_path)
	main_window.queue_free()


func apply_update_heightmap(mres:MResource):
	var ac = float(accuracy.text)
	var qtq = compress_qtq.button_pressed
	var fcompress = hfile_compress.selected
	var data = mres.get_heightmap_rf(false)
	mres.insert_heightmap_rf(data,ac,qtq,fcompress)

func apply_update_data(dname:StringName,mres:MResource):
	var state = data_state_final[dname]
	var fcompress = state["file_compress"]
	var compress = state["compress"]
	var format = state["format"]
	var data = mres.get_data(dname,false)
	mres.insert_data(data,dname,format,compress,fcompress)


func save_config_file():
	var ac = float(accuracy.text)
	var qtq = compress_qtq.button_pressed
	var hfcompress = hfile_compress.selected

	
	var path = terrain.dataDir.path_join(config_file_name)
	var conf := ConfigFile.new()
	conf.set_value("heightmap","accuracy",ac)
	conf.set_value("heightmap","compress_qtq",qtq)
	conf.set_value("heightmap","file_compress",hfcompress)
	
	var keys = data_state_final.keys()
	for key in keys:
		if key==&"heightmap":
			continue
		var state:Dictionary = data_state_final[key]
		conf.set_value(key,"file_compress",state["file_compress"])
		conf.set_value(key,"compress",state["compress"])
	conf.save(path)
