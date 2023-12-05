extends Node

@export var example = 99
@export_range(0, 100) var example_range = 100
@export_range(0, 100, 1) var example_range_step = 101
@export_range(0, 100, 1, "or_greater") var example_range_step_or_greater = 102

@export var color: Color
@export_color_no_alpha var color_no_alpha: Color
@export_node_path("Sprite2D", "Sprite3D", "Control", "Node") var nodepath := ^"hello"
@export var node: Node
@export var node_array: Array[Node]

func test():
	print(example)
	print(example_range)
	print(example_range_step)
	print(example_range_step_or_greater)
	print(color)
	print(color_no_alpha)
	print(nodepath)
	print(node)
	print(var_to_str(node_array))
