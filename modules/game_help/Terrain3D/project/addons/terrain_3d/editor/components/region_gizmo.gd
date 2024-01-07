extends EditorNode3DGizmo
	
var material: StandardMaterial3D
var selection_material: StandardMaterial3D
var region_position: Vector2
var region_size: float
var grid: Array[Vector2i]
var use_secondary_color: bool = false
var show_rect: bool = true

var main_color: Color = Color.GREEN_YELLOW
var secondary_color: Color = Color.RED
var grid_color: Color = Color.WHITE
var border_color: Color = Color.BLUE


func _init() -> void:
	material = StandardMaterial3D.new()
	material.set_flag(BaseMaterial3D.FLAG_DISABLE_DEPTH_TEST, true)
	material.set_flag(BaseMaterial3D.FLAG_ALBEDO_FROM_VERTEX_COLOR, true)
	material.set_shading_mode(BaseMaterial3D.SHADING_MODE_UNSHADED)
	material.set_albedo(Color.WHITE)
	
	selection_material = material.duplicate()
	selection_material.set_render_priority(0)


func _redraw() -> void:
	clear()
	
	var rect_position = region_position * region_size

	if show_rect:
		var modulate: Color = main_color if !use_secondary_color else secondary_color
		if abs(region_position.x) > 8 or abs(region_position.y) > 8:
			modulate = Color.GRAY
		draw_rect(Vector2(region_size,region_size)*.5 + rect_position, region_size, selection_material, modulate)
	
	for pos in grid:
		var grid_tile_position = Vector2(pos) * region_size
		if show_rect and grid_tile_position == rect_position:
			# Skip this one, otherwise focused region borders are not always visible due to draw order
			continue
			
		draw_rect(Vector2(region_size,region_size)*.5 + grid_tile_position, region_size, material, grid_color)
		
	draw_rect(Vector2.ZERO, region_size * 16.0, material, border_color)


func draw_rect(p_pos: Vector2, p_size: float, p_material: StandardMaterial3D, p_modulate: Color) -> void:
	var lines: PackedVector3Array = [
		Vector3(-1, 0, -1),
		Vector3(-1, 0, 1),
		Vector3(1, 0, 1),
		Vector3(1, 0, -1),
		Vector3(-1, 0, 1),
		Vector3(1, 0, 1),
		Vector3(1, 0, -1),
		Vector3(-1, 0, -1),
	]
	
	for i in lines.size():
		lines[i] = ((lines[i] / 2.0) * p_size) + Vector3(p_pos.x, 0, p_pos.y)
	
	add_lines(lines, p_material, false, p_modulate)
		
