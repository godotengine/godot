extends RefCounted # TODO: Fix standalone annotations parsing.

# GH-73843
@export_group("Resource")

# GH-78252
@export var prop_1: int
@export_category("prop_1")
@export var prop_2: int

func test():
	var resource := Resource.new()
	prints("Not shadowed:", resource.get_class())

	for property in get_property_list():
		if property.name in ["prop_1", "prop_2"]:
			print(property)
