@tool
extends BoxContainer

@onready var active_btn:=$active
@onready var segment_select_option:=$segment_select
@onready var socket_select_option:=$socket_select
var selection:EditorSelection

var obj:MCurveMesh

var instance_rid:RID
var scenario:RID

var active_root_scene:Node
var sockets_gizmos:Array
var selected_socket:int = -1


func _init():
	instance_rid = RenderingServer.instance_create()
	var ed := EditorScript.new()
	selection = ed.get_editor_interface().get_selection()
	selection.selection_changed.connect(selection_changed)

func _enter_tree():
	update_scenario()

func _exit_tree():
	if instance_rid.is_valid():
		RenderingServer.free_rid(instance_rid)

func set_curve_mesh(input:MCurveMesh):
	if not input and is_active():
		return
	obj = input
	udpate_sockets_gizmos()
	socket_select_option.clear()
	segment_select_option.clear()
	if not obj or not active_btn.button_pressed:
		visible = obj != null
		set_process(false)
		update_mesh()
		return
	visible = true
	set_process(true)
	active_root_scene = get_root_scene()
	for i in range(0,obj.intersections.size()):
		var seg:MIntersection = obj.intersections[i]
		segment_select_option.add_item("Seg "+str(i),i)
	if obj.intersections.size() >= 1:
		_on_segment_select_item_selected(0)
		segment_select_option.select(0)
	update_scenario()
	update_mesh()

func _on_segment_select_item_selected(index):
	udpate_sockets_gizmos()
	socket_select_option.clear()
	update_mesh()
	if index >= obj.intersections.size(): return
	var seg:MIntersection = obj.intersections[index]
	if not seg: return
	var socket = seg.sockets
	for i in range(0,socket.size()):
		socket_select_option.add_item("Socket "+str(i))
	if socket.size() > 0:
		socket_select_option.select(0)
		_on_socket_select_item_selected(0)

func _on_socket_select_item_selected(index):
	if index < 0 or index >= sockets_gizmos.size():
		return
	selected_socket = index
	selection.clear()
	selection.add_node(sockets_gizmos[index])

func _on_active_toggled(button_pressed):
	set_curve_mesh(obj)
	segment_select_option.visible = button_pressed
	socket_select_option.visible = button_pressed
	if not button_pressed and obj:
		selection.clear()
		selection.add_node(obj)

func update_mesh():
	RenderingServer.instance_set_base(instance_rid,RID())
	if not obj: return
	var seg_index:int = segment_select_option.selected
	if seg_index >= obj.intersections.size() or seg_index < 0: return
	var seg:MIntersection = obj.intersections[seg_index]
	if not seg: return
	var mesh_lod:MMeshLod = seg.mesh
	if not mesh_lod: return
	var mesh:Mesh = null
	for m in mesh_lod.meshes:
		if m:
			mesh = m
			break
	if not mesh: return
	RenderingServer.instance_set_base(instance_rid,mesh.get_rid())

func get_root_scene()->Node:
	var ed = EditorScript.new()
	var scene = ed.get_scene()
	return scene

func update_scenario():
	var scene = get_root_scene()
	if not scene:return
	if scene is Node3D:
		var w3d = scene.get_world_3d()
		if w3d:
			RenderingServer.instance_set_scenario(instance_rid,w3d.scenario)

func udpate_sockets_gizmos():
	for s in sockets_gizmos:
		s.queue_free()
	sockets_gizmos.clear()
	if not active_btn.button_pressed: return
	if not obj: return
	var seg_index:int = segment_select_option.selected
	if seg_index >= obj.intersections.size(): return
	var seg:MIntersection = obj.intersections[seg_index]
	if not seg: return
	var ed = EditorScript.new()
	var scene = ed.get_scene()
	if not scene:return
	for socket in seg.sockets:
		var marker = Marker3D.new()
		scene.add_child(marker)
		marker.global_transform = socket
		sockets_gizmos.push_back(marker)

func is_active():
	return active_btn.button_pressed

func _on_visibility_changed():
	if active_btn and not visible:
		active_btn.button_pressed = false

func selection_changed():
	if not is_active(): return
	if selected_socket < 0 or selected_socket >= sockets_gizmos.size(): return
	if active_root_scene != get_root_scene():
		active_btn.button_pressed = false
		set_curve_mesh(null)
		return
	var socket_marker = sockets_gizmos[selected_socket]
	var sel := selection.get_selected_nodes()
	if sel.size() > 1:
		selection.clear()
		selection.add_node(socket_marker)
		return
	if sel.size() == 1:
		if sel[0] != socket_marker:
			selection.clear()
			selection.add_node(socket_marker)
	if sel.size() == 0:
		selection.clear()
		selection.add_node(socket_marker)

func _process(delta):
	if not obj: return
	var seg_index = segment_select_option.get_selected_id()
	var socket_index = socket_select_option.get_selected_id()
	if socket_index < 0 or socket_index >= sockets_gizmos.size(): return
	var marker = sockets_gizmos[socket_index]
	var t:Transform3D = marker.global_transform
	obj.intersections[seg_index].sockets[socket_index] = t
