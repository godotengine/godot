@tool
extends VBoxContainer

@onready var window := $"../.."
@onready var hcontainer := $hcontainer
@onready var dcontainer := $dcontainer
@onready var import_error := $import_err
@onready var source_line := $HBoxContainer/source_line
@onready var save_line := $HBoxContainer2/LineEdit
@onready var file_dialog_source  := $"../../FileDialog_res_source"
@onready var file_dialog_save  := $"../../FileDialog_res_save"
@onready var dname_option := $dcontainer/dnames_option
@onready var dfile_compress_option := $dcontainer/dfile_compress
@onready var data_compress_option := $dcontainer/data_compress_option

@onready var compress_qtq_checkbox := $hcontainer/compress_qtq_checkbox
@onready var accuracy_line := $hcontainer/HBoxContainer3/accuracy_line
@onready var hfile_compress_option := $hcontainer/file_compress

const config_file_name := ".save_config.ini"

var dnames:PackedStringArray
var reg:RegEx
var save_config:Dictionary

func _init():
	reg = RegEx.new()
	var pattern :="(.*)_x\\d+_y\\d+\\.res"
	reg.compile(pattern)

func _on_source_dir_button_up():
	file_dialog_source.visible = true

func _on_save_dir_button_up():
	file_dialog_save.visible = true

func _on_file_dialog_res_source_dir_selected(dir):
	source_line.text = dir
	_on_source_line_text_changed(dir)

func _on_file_dialog_res_save_dir_selected(dir):
	save_line.text = dir

func _on_source_line_text_changed(new_text:String):
	dnames.clear()
	save_config.clear()
	hcontainer.visible = false
	dcontainer.visible = false
	if not new_text.is_absolute_path():
		perr("Is not an absoulute path")
		return
	if not DirAccess.dir_exists_absolute(new_text):
		perr("Dir does not exist")
		return
	var dir := DirAccess.open(new_text)
	if not dir:
		perr("Can not open Dir")
		return
	if dir.list_dir_begin() != OK:
		perr("Can not get file inside directory")
		return
	var file:String = dir.get_next()
	while  file != "":
		var m := reg.search(file)
		if m:
			if not dnames.has(m.strings[1]):
				dnames.push_back(m.strings[1])
		file = dir.get_next()
	if dnames.is_empty():
		perr("Can not find any Godot res file here")
		return
	for dname in dnames:
		if dname == "heightmap":
			hcontainer.visible = true
		else:
			dcontainer.visible = true
			save_config[dname] = {"compress":0,"file_compress":0}
			dname_option.add_item(dname)
			
	perr("")

func _on_data_compress_option_item_selected(index):
	if dname_option.selected < 0:
		return
	var dn:String = dname_option.get_item_text(dname_option.selected)
	var conf :Dictionary = save_config[dn]
	conf["compress"] = index
	save_config[dn] = conf

func _on_dfile_compress_item_selected(index):
	if dname_option.selected < 0:
		return
	var dn:String = dname_option.get_item_text(dname_option.selected)
	var conf :Dictionary = save_config[dn]
	conf["file_compress"] = index
	save_config[dn] = conf

func _on_dnames_option_item_selected(index):
	if dname_option.selected < 0:
		return
	var dn:String = dname_option.get_item_text(index)
	var conf :Dictionary = save_config[dn]
	dfile_compress_option.select(conf["file_compress"])
	data_compress_option.select(conf["compress"])


func _on_import_res_btn_button_up():
	
	var save_path:String = save_line.text
	if save_path.is_empty():
		perr("Save path is empty")
		return
	if not save_path.is_absolute_path():
		perr("Save path is not an absolute path")
		return
	if not DirAccess.dir_exists_absolute(save_path):
		perr("Save path does not exist")
		return
	
	var source_path:String = source_line.text
	
	#if save_config.is_empty():
	#	perr("Save config is empty, Please select a directory with Godot res exported with MTerrain")
	#	return
	if dnames.is_empty():
		perr("No data detectet in this Directory, Please select a directory with Godot res exported with MTerrain")
		return
	save_config_file()
	for dn in dnames:
		import_res(source_path,save_path,dn)
	window.queue_free()

func import_res(source_path:String,save_path:String,dname:String):
	var compress_qtq :bool= compress_qtq_checkbox.button_pressed
	var ac:float = float(accuracy_line.text)
	var hfile_compress:int= hfile_compress_option.selected
	var compress:int
	var dfile_compress:int
	if dname != "heightmap":
		var conf = save_config[dname]
		compress = conf["compress"]
		dfile_compress = conf["file_compress"]
	
	var x:int=0
	var y:int=0
	var find = false
	while true:
		var opath:String = source_path.path_join(dname+"_x"+str(x)+"_y"+str(y)+".res")
		if not ResourceLoader.exists(opath):
			y += 1
			x = 0
			opath = source_path.path_join(dname+"_x"+str(x)+"_y"+str(y)+".res")
			if not ResourceLoader.exists(opath):
				if not find:
					perr("Can not find "+dname+"_x0_y0.res")
				break
		find = true
		var ires = ResourceLoader.load(opath)
		if not (ires is Image):
			perr(opath+" is not an Image type")
			continue
		var spath:String = save_path.path_join("x"+str(x)+"_y"+str(y)+".res")
		var mres
		if ResourceLoader.exists(spath):
			mres = ResourceLoader.load(spath)
			if not (mres is MResource):
				perr("In save dir "+spath+" is not a MResource type")
				continue
		else:
			mres = MResource.new()
		if dname == "heightmap":
			mres.insert_heightmap_rf(ires.get_data(),ac,compress_qtq,hfile_compress)
		else:
			mres.insert_data(ires.get_data(),dname,ires.get_format(),compress,dfile_compress)
		ResourceSaver.save(mres,spath)
		x+=1

func save_config_file():
	var compress_qtq :bool= compress_qtq_checkbox.button_pressed
	var ac:float = float(accuracy_line.text)
	var hfile_compress:int= hfile_compress_option.selected

	
	var path = save_line.text.path_join(config_file_name)
	var conf := ConfigFile.new()
	conf.set_value("heightmap","accuracy",ac)
	conf.set_value("heightmap","compress_qtq",compress_qtq)
	conf.set_value("heightmap","file_compress",hfile_compress)
	
	var keys = save_config.keys()
	for key in keys:
		if key==&"heightmap":
			continue
		var state:Dictionary = save_config[key]
		conf.set_value(key,"file_compress",state["file_compress"])
		conf.set_value(key,"compress",state["compress"])
	conf.save(path)


func perr(msg:String):
	if msg.is_empty():
		import_error.visible = false
		return
	import_error.visible = true
	import_error.text = msg












