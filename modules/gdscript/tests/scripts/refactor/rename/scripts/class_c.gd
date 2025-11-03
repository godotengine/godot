class_name ClassCisB
extends B

const EnumClass = preload("enum.gd")


static func get_new_me() -> ClassCisB:
	return ClassCisB.new()


func _ready() -> void:
	print(EnumClass.ENUM_VALUE_1.ENUM_VALUE_1)
