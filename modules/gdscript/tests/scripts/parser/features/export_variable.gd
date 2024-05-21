extends Node

const Utils = preload("../../utils.notest.gd")

# Built-in types.
@export var test_weak_int = 1
@export var test_hard_int: int = 2
@export_range(0, 100) var test_range = 100
@export_range(0, 100, 1) var test_range_step = 101
@export_range(0, 100, 1, "or_greater") var test_range_step_or_greater = 102
@export var test_color: Color
@export_color_no_alpha var test_color_no_alpha: Color
@export_node_path("Sprite2D", "Sprite3D", "Control", "Node") var test_node_path := ^"hello"

# Enums.
@export var test_side: Side
@export var test_atm: AutoTranslateMode

# Resources and nodes.
@export var test_image: Image
@export var test_timer: Timer

# Arrays.
@export var test_array: Array
@export var test_array_bool: Array[bool]
@export var test_array_array: Array[Array]
@export var test_array_side: Array[Side]
@export var test_array_atm: Array[AutoTranslateMode]
@export var test_array_image: Array[Image]
@export var test_array_timer: Array[Timer]

# `@export_storage`.
@export_storage var test_storage_untyped
@export_storage var test_storage_weak_int = 3 # Property info still `Variant`, unlike `@export`.
@export_storage var test_storage_hard_int: int = 4

# `@export_custom`.
# NOTE: `PROPERTY_USAGE_NIL_IS_VARIANT` flag will be removed.
@export_custom(PROPERTY_HINT_ENUM, "A,B,C") var test_export_custom_untyped
@export_custom(PROPERTY_HINT_ENUM, "A,B,C") var test_export_custom_weak_int = 5
@export_custom(PROPERTY_HINT_ENUM, "A,B,C") var test_export_custom_hard_int: int = 6

func test():
	for property in get_property_list():
		if str(property.name).begins_with("test_"):
			Utils.print_property_extended_info(property, self)
