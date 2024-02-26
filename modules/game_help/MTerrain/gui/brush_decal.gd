@tool
extends Decal
class_name MBrushDecal

var radius:float=5.0

func set_brush_size(input:float):
	size.x = input
	size.z = input
	radius = input/2

func get_brush_size()->float:
	return size.x

func change_brush_color(input:Color):
	modulate = input
