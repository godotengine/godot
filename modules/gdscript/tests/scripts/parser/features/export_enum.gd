@export_enum("Red", "Green", "Blue") var untyped

@export_enum("Red", "Green", "Blue") var weak_int = 0
@export_enum("Red", "Green", "Blue") var weak_string = ""

@export_enum("Red", "Green", "Blue") var hard_int: int
@export_enum("Red", "Green", "Blue") var hard_string: String

@export_enum("Red:10", "Green:20", "Blue:30") var with_values

func test():
	for property in get_property_list():
		if property.name in ["untyped", "weak_int", "weak_string", "hard_int",
				"hard_string", "with_values"]:
			prints(property.name, property.type, property.hint_string)
