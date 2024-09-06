@tool
extends Window

signal keymap_changed
signal restore_default_keymap_requested

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

@onready var save_config = find_child("Manage Images")

var version:String

var terrain:MTerrain=null
var region_meter_size:int=0
var terrain_area:int
var region_pixel_size:int=0
var region_grid_size:Vector2i
var region_count:int
var terrain_meter_size:Vector2
var terrain_pixel_size:Vector2

var mtools

func generate_info(_t:MTerrain,_version:String, keyboard_actions):
	save_config.init_save_config(_t)
	terrain = _t
	version = _version
	if not _t:
		Terror.text = "No Active Terrain"
		Terror.visible = true
		return
	var errstr:String=""
	if terrain.terrain_quad_count.x%terrain.region_quad_count!=0 or terrain.terrain_quad_count.x%terrain.region_quad_count!=0:
		errstr="Terrain size in x or y direction must be divisible by region size\nfor example in this case as your region size %s your terain size can be (%d,%d) or (%d,%d) or (%d,%d)\nThese are in grid unit which currently is %d meter"
		var ex1:=Vector2i(terrain.region_quad_count*1,terrain.region_quad_count*1)
		var ex2:=Vector2i(terrain.region_quad_count*2,terrain.region_quad_count*2)
		var ex3:=Vector2i(terrain.region_quad_count*1,terrain.region_quad_count*2)
		errstr = errstr % [terrain.region_quad_count,ex1.x,ex1.y,ex2.x,ex2.y,ex3.x,ex3.y,terrain.get_base_size()]
	if not errstr.is_empty():
		errstr +="\n\nChange these and close and reopen this window"
		Terror.text = errstr
		Terror.visible = true
	region_meter_size = (terrain.get_base_size()*terrain.region_quad_count)
	region_pixel_size = (region_meter_size/terrain.get_h_scale()) + 1
	terrain_meter_size = terrain.terrain_quad_count*terrain.get_base_size()
	terrain_area = terrain_meter_size.x * terrain_meter_size.x
	terrain_pixel_size = (terrain_meter_size/terrain.get_h_scale()) + Vector2(1,1)
	region_grid_size = (terrain.terrain_quad_count) / terrain.region_quad_count
	region_count = region_grid_size.x * region_grid_size.y
	var warnstr:String=""
	warnstr = "Images in data directory should be width=height=%d one common edge pixel between regions will be created at load time."%(region_pixel_size-1)
	Rwarning.text = warnstr
	tsizeg.text += " %d X %d" % [terrain.terrain_quad_count.x,terrain.terrain_quad_count.y]
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
	create_keymapping_interface(keyboard_actions)
	
func create_keymapping_interface(keyboard_actions):
	var mterrain_actions_list = find_child("mterrain_action_list")
	for child in mterrain_actions_list.get_children():
		child.queue_free()
		mterrain_actions_list.remove_child(child)
	var mpath_actions_list = find_child("mpath_action_list")
	for child in mpath_actions_list .get_children():
		child.queue_free()
		mterrain_actions_list.remove_child(child)
	var restore_default_keymap = find_child("restore_default_keymap")
	restore_default_keymap.pressed.connect( func(): restore_default_keymap_requested.emit() )
	for action in keyboard_actions:
		var item = preload("res://addons/m_terrain/gui/mtools_keyboard_shortcut_item.tscn").instantiate()				
		if action.name.begins_with("mterrain_"):			
			mterrain_actions_list.add_child(item)
			item.label.text = action.name.substr(9)
			item.name = action.name
		elif action.name.begins_with("mpath_"):			
			mpath_actions_list.add_child(item)
			item.label.text = action.name.substr(6)
			item.name = action.name
		item.value.text = OS.get_keycode_string(action.keycode)
		item.keymap_changed.connect( func(who, keycode, ctrl,alt,shift): keymap_changed.emit(who, keycode,ctrl,alt,shift))

func _on_info_meta_clicked(meta):
	OS.shell_open(meta)

func _on_close_requested():
	queue_free()

func _on_delete_uniform_pressed():
	var confirm_label = find_child("delete_confirm_label")
	if confirm_label.visible:
		mtools.remove_image(mtools.get_active_mterrain(), find_child("data_name_option").text)
		save_config.init_save_config(terrain)	
		confirm_label.visible = false
	else:
		confirm_label.visible = true
