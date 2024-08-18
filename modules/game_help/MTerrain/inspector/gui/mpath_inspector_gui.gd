@tool
extends Control


@onready var point_count_lable:=$point_count_lable

var is_started:=false

var gizmo:EditorNode3DGizmoPlugin

var current_path:MPath
var current_curve:MCurve
var current_curve_mesh:MCurveMesh=null

var connection_mode :=true
var mesh_mode:=true

var child_selector:OptionButton
var items:ItemList
var connection_tab:Button
var intersection_tab:Button
func start():
	if is_started: return
	is_started = true
	child_selector = $child_selctor
	items = $itemlist
	items.item_selected.connect(Callable(self,"item_selected"))
	connection_tab = $HBoxContainer/connection_tab
	intersection_tab = $HBoxContainer/intersection_tab
	connection_tab.connect("pressed",Callable(self,"change_tab").bind(true))
	intersection_tab.connect("pressed",Callable(self,"change_tab").bind(false))
	$HBoxContainer2/clear_override.connect("button_up",Callable(self,"clear_override"))
	$HBoxContainer2/Remove_mesh.connect("button_up",Callable(self,"remove_mesh"))
	$mesh_mode_option.connect("item_selected",Callable(self,"mesh_mode_selected"))
	if not gizmo: printerr("Gizmo is NULL")
	gizmo.connect("selection_changed",Callable(self,"update_curve_item_selection"))

func set_path(input:MPath)->void:
	start()
	current_path = input
	current_curve = current_path.curve
	child_selector.clear()
	child_selector.add_item("ovveride")
	if not input: return
	var children = input.get_children()
	for child in children:
		if child is MCurveMesh:
			child_selector.add_item(child.name)

func _on_child_selctor_item_selected(index):
	set_curve_mesh_gui_visible(false)
	if index==0 or not current_path:
		return
	var cname = child_selector.get_item_text(index)
	var child = current_path.get_node(cname)
	if not child: return
	if child is MCurveMesh:
		set_curve_mesh_gui_visible(true)
		current_curve_mesh = child
		update_curve_mesh_items()

func set_curve_mesh_gui_visible(input:bool):
	$HBoxContainer.visible = input
	$HBoxContainer2.visible = input
	$mesh_mode_option.visible = input
	$itemlist.visible = input

func change_tab(_conn_mod:bool):
	connection_tab.button_pressed = _conn_mod
	intersection_tab.button_pressed = not _conn_mod
	if _conn_mod == connection_mode: return
	connection_mode = _conn_mod
	update_curve_mesh_items()

func update_curve_mesh_items():
	if not current_curve_mesh: return
	items.clear()
	var ed:=EditorScript.new()
	var preview:EditorResourcePreview= ed.get_editor_interface().get_resource_previewer()
	var count:int = 0
	if not mesh_mode:
		for mat in current_curve_mesh.materials:
			if not mat:
				items.add_item("empty")
				continue
			var mname:String = mat.get_path().get_file().get_basename()
			if mname.is_empty():
				mname = str(count)
			items.add_item(mname)
			preview.queue_edited_resource_preview(mat,self,"set_icon",count)
			count+=1
		update_curve_item_selection()
		return
	if connection_mode:count = current_curve_mesh.meshes.size()
	else:count = current_curve_mesh.intersections.size()
	for i in range(count):
		var mlod:MMeshLod
		if connection_mode: mlod = current_curve_mesh.meshes[i]
		else: mlod = current_curve_mesh.intersections[i].mesh
		if not mlod:
			items.add_item("empty")
			continue
		var m:Mesh = mlod.meshes[0]
		if not m:
			items.add_item("empty")
			continue
		var mname:String = m.get_path().get_file().get_basename()
		if mname.is_empty():
			mname = str(i)
		items.add_item(mname)
		preview.queue_edited_resource_preview(m,self,"set_icon",i)
	update_curve_item_selection()

# must be called after update_curve_mesh_items()
func update_curve_item_selection():
	if not current_curve_mesh or not gizmo: return
	var ids:PackedInt64Array = get_selected_ids()
	var ov_index:int=-100 # some invalide number
	for cid in ids:
		var current_index:int
		if mesh_mode: current_index = current_curve_mesh.override.get_mesh_override(cid)
		else: current_index = current_curve_mesh.override.get_material_override(cid)
		if ov_index == -100: ov_index = current_index
		if current_index != ov_index: ## multiple connection is selected with multiple ovverride value
			ov_index = -1
			break
		### Up to this point connection ovveride is same
		ov_index = current_index
	$HBoxContainer2/Remove_mesh.button_pressed = connection_mode and ov_index == -2
	if ov_index < 0:
		items.deselect_all()
	else:
		items.select(ov_index)

func set_icon(path:String,preview:Texture2D,thumnail_preview:Texture2D,index:int):
	items.set_item_icon(index,preview)

func item_selected(index:int):
	if not is_instance_valid(current_curve_mesh) or not gizmo: return
	var ids:PackedInt64Array = get_selected_ids()
	for cid in ids:
		if mesh_mode:
			current_curve_mesh.override.set_mesh_override(cid,index)
		else:
			current_curve_mesh.override.set_material_override(cid,index)
	update_curve_item_selection()

func clear_override():
	if not is_instance_valid(current_curve_mesh) or not gizmo: return
	var ids:PackedInt64Array = get_selected_ids()
	if mesh_mode:
		for cid in ids:
			current_curve_mesh.override.clear_mesh_override(cid)
	else:
		for cid in ids:
			current_curve_mesh.override.clear_material_override(cid)
	update_curve_item_selection()

func remove_mesh():
	if not is_instance_valid(current_curve_mesh) or not gizmo: return
	var ids:PackedInt64Array = get_selected_ids()
	
	for cid in ids:
		current_curve_mesh.override.set_mesh_override(cid,-2)
	update_curve_item_selection()

func get_selected_ids()->PackedInt64Array:
	if connection_mode:
		return gizmo.get_selected_connections()
	else:
		return gizmo.get_selected_points64()

func mesh_mode_selected(index:int):
	mesh_mode = index == 0
	update_curve_mesh_items()


func _on_update_info_timer_timeout():
	var point_count:int = 0
	if current_curve:
		point_count = current_curve.get_points_count()
	point_count_lable.text = "Point count " + str(point_count)
