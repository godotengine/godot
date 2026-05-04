func print_value(value: Variant) -> void:
	if value is Object:
		@warning_ignore("unsafe_method_access")
		print("<%s>" % value.get_class())
	else:
		print(var_to_str(value))

func test():
	var int_value := 1
	print_value(int_value as Variant)
	print_value(int_value as int)
	print_value(int_value as float)

	print("---")

	var node_value := Node.new()
	print_value(node_value as Variant)
	print_value(node_value as Object)
	print_value(node_value as Node)
	print_value(node_value as Node2D)
	node_value.free()

	print("---")

	var null_value = null
	print_value(null_value as Variant)
	@warning_ignore("unsafe_cast")
	print_value(null_value as Node)

	print("---")

	var string_array: Array[String] = ["abc", "123"]
	print_value(string_array as Variant)
	print_value(string_array as Array)
	print_value(string_array as Array[String])
	print_value(string_array as PackedStringArray)
	print_value(string_array as PackedInt32Array)
	print_value(string_array as PackedVector2Array)

	print("---")

	var packed_string_array: PackedStringArray = ["abc", "123"]
	print_value(packed_string_array as Variant)
	print_value(packed_string_array as Array)
	print_value(packed_string_array as Array[String])
	print_value(packed_string_array as PackedStringArray)

	print("---")

	var array: Array = [1, "a"]
	var int_array: Array[int] = [1]
	var vector2_array: Array[Vector2] = [Vector2.ONE]
	print_value(array as PackedStringArray)
	print_value(int_array as PackedStringArray)
	print_value(vector2_array as PackedStringArray)

	print("---")

	var string_dict: Dictionary[String, String] = {}
	print_value(string_dict as Variant)
	print_value(string_dict as Dictionary)
	print_value(string_dict as Dictionary[String, String])
