@tool
extends Window

@onready var Terror:=$base/TabContainer/Terrain/VBoxContainer/Error
@onready var Rwarning:=$base/TabContainer/Region/VBoxContainer/Warning
@onready var tsizeg:=$base/TabContainer/Terrain/VBoxContainer/tsizeg
@onready var tsizem:=$base/TabContainer/Terrain/VBoxContainer/tsizem
@onready var takm:=$base/TabContainer/Terrain/VBoxContainer/takm
@onready var tsizepx:=$base/TabContainer/Terrain/VBoxContainer/tsizepx
@onready var tregioncount:=$base/TabContainer/Terrain/VBoxContainer/tregioncount
@onready var tregioncountX:=$base/TabContainer/Terrain/VBoxContainer/tregioncountX
@onready var tregioncountY:=$base/TabContainer/Terrain/VBoxContainer/tregioncountZ

@onready var rsizem:=$base/TabContainer/Region/VBoxContainer/rsizem
@onready var rsizepixel:=$base/TabContainer/Region/VBoxContainer/rsizepixel

@onready var base_unit:=$"base/TabContainer/Base Size/base_unit"

@onready var info:=$base/TabContainer/info/info

@onready var save_config:=$"base/TabContainer/Save config"

var version:String

var terrain:MTerrain=null
var region_meter_size:int=0
var terrain_area:int
var region_pixel_size:int=0
var region_grid_size:Vector2i
var region_count:int
var terrain_meter_size:Vector2
var terrain_pixel_size:Vector2



func generate_info(_t:MTerrain,_version:String):
	save_config.init_save_config(_t)
	terrain = _t
	version = _version
	if not _t:
		Terror.text = "No Active Terrain"
		Terror.visible = true
		return
	var errstr:String=""
	if terrain.terrain_size.x%terrain.region_size!=0 or terrain.terrain_size.x%terrain.region_size!=0:
		errstr="Terrain size in x or y direction must be divisible by region size\nfor example in this case as your region size %s your terain size can be (%d,%d) or (%d,%d) or (%d,%d)\nThese are in grid unit which currently is %d meter"
		var ex1:=Vector2i(terrain.region_size*1,terrain.region_size*1)
		var ex2:=Vector2i(terrain.region_size*2,terrain.region_size*2)
		var ex3:=Vector2i(terrain.region_size*1,terrain.region_size*2)
		errstr = errstr % [terrain.region_size,ex1.x,ex1.y,ex2.x,ex2.y,ex3.x,ex3.y,terrain.get_base_size()]
	if not errstr.is_empty():
		errstr +="\n\nChange these and close and reopen this window"
		Terror.text = errstr
		Terror.visible = true
	region_meter_size = (terrain.get_base_size()*terrain.region_size)
	region_pixel_size = (region_meter_size/terrain.get_h_scale()) + 1
	terrain_meter_size = terrain.terrain_size*terrain.get_base_size()
	terrain_area = terrain_meter_size.x * terrain_meter_size.x
	terrain_pixel_size = (terrain_meter_size/terrain.get_h_scale()) + Vector2(1,1)
	region_grid_size = (terrain.terrain_size) / terrain.region_size
	region_count = region_grid_size.x * region_grid_size.y
	var warnstr:String=""
	warnstr = "Images in data directory should be width=height=%d one common edge pixel between regions will be created at load time."%(region_pixel_size-1)
	Rwarning.text = warnstr
	tsizeg.text += " %d X %d" % [terrain.terrain_size.x,terrain.terrain_size.y]
	tsizem.text += " %dm X %dm" % [terrain_meter_size.x,terrain_meter_size.y]
	takm.text += " %10.3f km2" % [float(terrain_area)/1000000.0]
	tsizepx.text += " %d X %d" % [terrain_pixel_size.x,terrain_pixel_size.y]
	tregioncount.text += " %d" % [region_count]
	tregioncountX.text += " %d" % [region_grid_size.x]
	tregioncountY.text += " %d" % [region_grid_size.y]
	rsizem.text += " %dm" % [region_meter_size]
	rsizepixel.text += " %d" % [region_pixel_size]
	var vc = (terrain.get_base_size()/terrain.get_h_scale())+1
	vc *= vc
	base_unit.text = base_unit.text % [terrain.get_base_size(),vc]
	info.text = info.text % version

func _on_info_meta_clicked(meta):
	OS.shell_open(meta)

func _on_close_requested():
	queue_free()




