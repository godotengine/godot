extends EditorInspectorPlugin

var gizmo:EditorNode3DGizmoPlugin

var gui_res = preload("res://addons/m_terrain/inspector/gui/mpath_inspector_gui.tscn")
var gui
var curve_terrain_res = preload("res://addons/m_terrain/inspector/gui/curve_terrain.tscn")
var curve_terrain:Control

func _can_handle(object):
	return object is MPath

func _parse_begin(object):
	if not(object is MPath) or not object or not object.curve:
		return
	if not gui:
		gui = gui_res.instantiate()
	add_custom_control(gui)
	gui.gizmo = gizmo
	gui.set_path(object)
	## Curve terrain
	if gizmo.tools.get_active_mterrain():
		if not curve_terrain:
			curve_terrain = curve_terrain_res.instantiate()
		add_custom_control(curve_terrain)
		gizmo.curve_terrain.terrain = gizmo.tools.get_active_mterrain()
		curve_terrain.start_curve_terrain(gizmo,object)
		




