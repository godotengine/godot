extends Node

const Utils = preload("../../utils.notest.gd")

@export var test_weak_int = 1
@export var test_hard_int: int = 2
@export_storage var test_storage_untyped
@export_storage var test_storage_weak_int = 3 # Property info still `Variant`, unlike `@export`.
@export_storage var test_storage_hard_int: int = 4
@export_range(0, 100) var test_range = 100
@export_range(0, 100, 1) var test_range_step = 101
@export_range(0, 100, 1, "or_greater") var test_range_step_or_greater = 102
@export var test_color: Color
@export_color_no_alpha var test_color_no_alpha: Color
@export_node_path("Sprite2D", "Sprite3D", "Control", "Node") var test_node_path := ^"hello"
@export var test_node: Node
@export var test_node_array: Array[Node]

func test():
	for property in get_property_list():
		if str(property.name).begins_with("test_"):
			Utils.print_property_extended_info(property, self)
