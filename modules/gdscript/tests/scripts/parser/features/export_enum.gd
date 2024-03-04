const Utils = preload("../../utils.notest.gd")

@export_enum("Red", "Green", "Blue") var test_untyped

@export_enum("Red", "Green", "Blue") var test_weak_int = 0
@export_enum("Red", "Green", "Blue") var test_weak_string = ""

@export_enum("Red", "Green", "Blue") var test_hard_int: int
@export_enum("Red", "Green", "Blue") var test_hard_string: String

@export_enum("Red:10", "Green:20", "Blue:30") var test_with_values

func test():
	for property in get_property_list():
		if str(property.name).begins_with("test_"):
			Utils.print_property_extended_info(property, self)
