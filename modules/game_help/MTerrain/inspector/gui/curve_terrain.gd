@tool
extends VBoxContainer

var grass_icon_tex = preload("res://addons/m_terrain/icons/foilage_icon.png")

var curve_terrain:MCurveTerrain
var gizmo
var path:MPath
var active_grass:MGrass=null

func start_curve_terrain(_gizmo,_path):
	$tsv2/deform_tools.get_popup().connect("index_pressed",deform_tool_index_pressed)
	$tsv3/paint_tools.get_popup().connect("index_pressed",paint_tool_index_pressed)
	$g_container/gt/grass_tool.get_popup().connect("index_pressed",grass_tool_index_pressed)
	gizmo = _gizmo
	path = _path
	curve_terrain = gizmo.curve_terrain
	curve_terrain.curve = path.curve
	if path.has_meta("curve_terrain"):
		var d = path.get_meta("curve_terrain")
		set_values(d)
	else:
		set_default_values()
	### Grass system is a bit different
	load_grass_settings()
	

func set_default_values():
	#print("set default -----------------------")
	var new_mcurve:=MCurveTerrain.new()
	gizmo.auto_terrain_deform = false
	gizmo.auto_terrain_paint = false
	gizmo.auto_grass_modify = false
	var d = get_curve_terrain_data(new_mcurve)
	set_values(d)
	

func set_values(d:Dictionary):
	$panle_open.set_pressed_no_signal(d["panel_open"]) ; panle_open_toggled(d["panel_open"])
	$tsv2/auto_deform.set_pressed_no_signal(d["auto_deform"]) ; gizmo.auto_terrain_deform = d["auto_deform"]
	$tsv/tilt.set_pressed_no_signal(d["tilt"]) ; curve_terrain.apply_tilt = d["tilt"]
	$tsv/scale.set_pressed_no_signal(d["scale"]) ; curve_terrain.apply_scale = d["scale"]
	$tlv/tlayer.text = d["tlayer"] ; curve_terrain.terrain_layer_name = d["tlayer"]
	$rv/dradius.set_value_no_signal(d["dradius"]) ; curve_terrain.deform_radius = d["dradius"]
	$fv/dfalloff.set_value_no_signal(d["dfalloff"]) ; curve_terrain.deform_falloff = d["dfalloff"]
	$ov/doffset.set_value_no_signal(d["doffset"]) ; curve_terrain.deform_offest = d["doffset"]
	############# Paint
	$tsv3/auto_paint.set_pressed_no_signal(d["auto_paint"]) ; gizmo.auto_terrain_paint = d["auto_paint"]
	$ov2/iname.text = d["iname"] ; curve_terrain.terrain_image_name = d["iname"]
	$ov3/pradius.set_value_no_signal(d["pradius"]) ; curve_terrain.paint_radius = d["pradius"]
	$ov5/pfalloff.set_value_no_signal(d["pfalloff"]) ; curve_terrain.paint_falloff = d["pfalloff"]
	$ov4/pcolor.color = d["pcolor"] ; curve_terrain.paint_color = d["pcolor"]
	$ov4/bgcolor.color = d["bgcolor"] ; curve_terrain.bg_color = d["bgcolor"]
	$tsv2/deform_tools.set("popup/item_0/checked",d["donly_selected"])
	$tsv3/paint_tools.set("popup/item_0/checked",d["ponly_selected"])
	$g_container/gt/auto_g_modify.set_pressed_no_signal(d["auto_grass"]) ; gizmo.auto_grass_modify = d["auto_grass"]

func get_curve_terrain_data(ct:MCurveTerrain)->Dictionary:
	var d:Dictionary
	d["panel_open"] = gizmo.is_curve_terrain_panel_open
	d["auto_deform"] = gizmo.auto_terrain_deform
	d["tilt"] = ct.apply_tilt
	d["scale"] = ct.apply_scale
	d["tlayer"] = ct.terrain_layer_name
	d["doffset"] = ct.deform_offest
	d["dradius"] = ct.deform_radius
	d["dfalloff"] = ct.deform_falloff
	d["auto_paint"] = gizmo.auto_terrain_paint
	d["iname"] = ct.terrain_image_name
	d["pcolor"] = ct.paint_color
	d["bgcolor"] = ct.bg_color
	d["pradius"] = ct.paint_radius
	d["pfalloff"] = ct.paint_falloff
	d["donly_selected"] = $tsv2/deform_tools.get("popup/item_0/checked")
	d["ponly_selected"] = $tsv3/paint_tools.get("popup/item_0/checked")
	d["auto_grass"] = gizmo.auto_grass_modify
	return d

func update_meta_values():
	var d = get_curve_terrain_data(curve_terrain)
	path.set_meta("curve_terrain",d)

func _on_panle_open_toggled(toggled_on):
	var child_count = get_child_count()
	gizmo.is_curve_terrain_panel_open = toggled_on
	for c in range(2,child_count):
		get_child(c).visible = toggled_on
	update_meta_values()

func panle_open_toggled(toggled_on):
	var child_count = get_child_count()
	for c in range(2,child_count):
		get_child(c).visible = toggled_on

func _on_auto_deform_toggled(toggled_on):
	gizmo.auto_terrain_deform = toggled_on
	update_meta_values()

func _on_tilt_toggled(toggled_on):
	curve_terrain.apply_tilt = toggled_on
	update_meta_values()

func _on_scale_toggled(toggled_on):
	curve_terrain.apply_scale = toggled_on
	update_meta_values()

func _on_tlayer_text_changed(new_text):
	curve_terrain.terrain_layer_name = new_text
	update_meta_values()

func _on_dradius_value_changed(value):
	curve_terrain.deform_radius = value
	update_meta_values()

func _on_dfalloff_value_changed(value):
	curve_terrain.deform_falloff = value
	update_meta_values()

func _on_doffset_value_changed(value):
	curve_terrain.deform_offest = value
	update_meta_values()

func _on_auto_paint_toggled(toggled_on):
	gizmo.auto_terrain_paint = toggled_on
	update_meta_values()

func _on_iname_text_changed(new_text):
	curve_terrain.terrain_image_name = new_text
	update_meta_values()

func _on_pradius_value_changed(value):
	curve_terrain.paint_radius = value
	update_meta_values()

func _on_pfalloff_value_changed(value):
	curve_terrain.paint_falloff = value
	update_meta_values()

func _on_pcolor_color_changed(color):
	curve_terrain.paint_color = color
	update_meta_values()

func _on_bgcolor_color_changed(color):
	curve_terrain.bg_color = color
	update_meta_values()


func deform_tool_index_pressed(index:int):
	var only_selected:bool=$tsv2/deform_tools.get("popup/item_0/checked")
	if index==0:
		$tsv2/deform_tools.set("popup/item_0/checked",not only_selected)
		$tsv2/deform_tools.call_deferred("show_popup")
		update_meta_values()
		return
	if index==1:
		gizmo.deform(only_selected)
		return
	if index == 2:
		gizmo.clear_deform(only_selected)
		return
	if index == 3:
		gizmo.clear_deform_large(only_selected)
		return

func paint_tool_index_pressed(index:int):
	var only_selected:bool=$tsv3/paint_tools.get("popup/item_0/checked")
	if index==0:
		$tsv3/paint_tools.set("popup/item_0/checked",not only_selected)
		$tsv3/paint_tools.call_deferred("show_popup")
		update_meta_values()
		return
	if index==1:
		gizmo.paint(only_selected)
		return
	if index == 2:
		gizmo.clear_paint(only_selected)
		return
	if index == 3:
		gizmo.clear_paint_large(only_selected)
		return
	
func _on_auto_g_modify_toggled(toggled_on):
	gizmo.auto_grass_modify = toggled_on
	update_meta_values()

###### GRASS
func load_grass_settings():
	var grass_count:int = 0
	var terrain_children:Array = curve_terrain.terrain.get_children()
	var settings:Dictionary
	for grass in terrain_children:
		if not(grass is MGrass):
			continue
		grass_count+=1
		$g_container/grass_list.add_item(grass.name,grass_icon_tex)
		if grass.has_meta("curve_terrain"):
			settings[grass.name] = grass.get_meta("curve_terrain")
	set_grass_items_visibilty(grass_count!=0)
	gizmo.grass_modify_settings = settings
	if grass_count !=0 :
		$g_container/grass_list.select(0)
		_on_grass_list_item_selected(0)

func _on_grass_list_item_selected(index):
	### This because of gradius_value_changed and goffset_value_changed
	### their siginal conflict this signal
	### but making this call_deferred correct that
	### I hope this does not break in future
	call_deferred("change_active_grass",index)

func change_active_grass(index):
	var grass_name = $g_container/grass_list.get_item_text(index)
	if not curve_terrain.terrain.has_node(grass_name):
		printerr("Can not find "+grass_name)
		return
	active_grass = curve_terrain.terrain.get_node(grass_name)
	if not gizmo.grass_modify_settings.has(grass_name):
		## restoring default setting
		$g_container/gh/gactive.set_pressed_no_signal(false)
		$g_container/gh/gadd.set_pressed_no_signal(false)
		$g_container/ov7/gradius.set_value_no_signal(6.0)
		$g_container/ov6/goffset.set_value_no_signal(0.0)
		return
	var g_setting:Dictionary= gizmo.grass_modify_settings[grass_name]
	$g_container/gh/gactive.set_pressed_no_signal(g_setting["active"])
	$g_container/gh/gadd.set_pressed_no_signal(g_setting["add"])
	$g_container/ov7/gradius.set_value_no_signal(g_setting["radius"])
	$g_container/ov6/goffset.set_value_no_signal(g_setting["offset"])

func upate_grass_meta():
	if not active_grass:
		printerr("No active grass")
		return
	if not is_instance_valid(active_grass):
		printerr("Active grass is not valid")
		return
	var g_setting:Dictionary
	g_setting["active"] = $g_container/gh/gactive.button_pressed
	g_setting["add"] = $g_container/gh/gadd.button_pressed
	g_setting["radius"] = $g_container/ov7/gradius.value
	g_setting["offset"] = $g_container/ov6/goffset.value
	gizmo.grass_modify_settings[active_grass.name] = g_setting
	active_grass.set_meta("curve_terrain",g_setting)
	print(active_grass.name ," update offset ",g_setting["offset"])

func set_grass_items_visibilty(input:bool):
	## change the visibilty of nodes not container
	## to not collaps with the main terrain visbilty
	var childer:Array= $g_container.get_children()
	for c in childer:
		c.visible = input

func _on_gradius_value_changed(value):
	upate_grass_meta()

func _on_goffset_value_changed(value):
	upate_grass_meta()


func grass_tool_index_pressed(index:int):
	var only_selected:bool=$g_container/gt/grass_tool.get("popup/item_0/checked")
	if index==0:
		$g_container/gt/grass_tool.set("popup/item_0/checked",not only_selected)
		$g_container/gt/grass_tool.call_deferred("show_popup")
		update_meta_values()
		return
	if index==1:
		gizmo.modify_grass(only_selected)
		return
	if index == 2:
		gizmo.clear_grass(only_selected)
		return
	if index == 3:
		gizmo.clear_grass_large(only_selected)
		return


