extends "res://addons/terrain_3d/src/operation_builder.gd"


const MultiPicker: Script = preload("res://addons/terrain_3d/src/multi_picker.gd")


func _get_point_picker() -> MultiPicker:
	return tool_settings.settings["gradient_points"]


func _get_brush_size() -> float:
	return tool_settings.get_setting("size")


func _is_drawable() -> bool:
	return tool_settings.get_setting("drawable")


func is_picking() -> bool:
	return not _get_point_picker().all_points_selected()


func pick(p_global_position: Vector3, p_terrain: Terrain3D) -> void:
	if not _get_point_picker().all_points_selected():
		_get_point_picker().add_point(p_global_position)


func is_ready() -> bool:
	return _get_point_picker().all_points_selected() and not _is_drawable()


func apply_operation(p_editor: Terrain3DEditor, p_global_position: Vector3, p_camera_direction: float) -> void:
	var points: PackedVector3Array = _get_point_picker().get_points()
	assert(points.size() == 2)
	assert(not _is_drawable())
	
	var brush_size: float = _get_brush_size()
	assert(brush_size > 0.0)
	
	var start: Vector3 = points[0]
	var end: Vector3 = points[1]
	
	p_editor.start_operation(start)
	
	var dir: Vector3 = (end - start).normalized()
	
	var pos: Vector3 = start
	while dir.dot(end - pos) > 0.0:
		p_editor.operate(pos, p_camera_direction)
		pos += dir * brush_size * 0.2
	
	p_editor.stop_operation()
	
	_get_point_picker().clear()

