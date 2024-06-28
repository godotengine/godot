extends RefCounted


const ToolSettings: Script = preload("res://addons/terrain_3d/src/tool_settings.gd")


var tool_settings: ToolSettings


func is_picking() -> bool:
	return false


func pick(p_global_position: Vector3, p_terrain: Terrain3D) -> void:
	pass


func is_ready() -> bool:
	return false


func apply_operation(editor: Terrain3DEditor, p_global_position: Vector3, p_camera_direction: float) -> void:
	pass
