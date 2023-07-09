@export_enum("Red:10", "Green:20", "Blue:30") var with_values

var temp_array_variant: Array[Variant]
var temp_array_int: Array[int]
var temp_array_string: Array[String]
var temp_array_string_name: Array[StringName]

@export_enum("Red", "Green", "Blue") var weak_variant
@export_enum("Red", "Green", "Blue") var weak_int = 0
@export_enum("Red", "Green", "Blue") var weak_string = ""
@export_enum("Red", "Green", "Blue") var weak_string_name = &""
@export_enum("Red", "Green", "Blue") var weak_array = []
@export_enum("Red", "Green", "Blue") var weak_array_variant = temp_array_variant
@export_enum("Red", "Green", "Blue") var weak_array_int = temp_array_int
@export_enum("Red", "Green", "Blue") var weak_array_string = temp_array_string
@export_enum("Red", "Green", "Blue") var weak_array_string_name = temp_array_string_name

@export_enum("Red", "Green", "Blue") var hard_variant: Variant
@export_enum("Red", "Green", "Blue") var hard_int: int
@export_enum("Red", "Green", "Blue") var hard_string: String
@export_enum("Red", "Green", "Blue") var hard_string_name: StringName
@export_enum("Red", "Green", "Blue") var hard_array: Array
@export_enum("Red", "Green", "Blue") var hard_array_variant: Array[Variant]
@export_enum("Red", "Green", "Blue") var hard_array_int: Array[int]
@export_enum("Red", "Green", "Blue") var hard_array_string: Array[String]
@export_enum("Red", "Green", "Blue") var hard_array_string_name: Array[StringName]

func test():
	for property in get_property_list():
		if property.usage & PROPERTY_USAGE_SCRIPT_VARIABLE and not str(property.name).begins_with("temp_"):
			prints(property.name, property.type, property.hint_string)
