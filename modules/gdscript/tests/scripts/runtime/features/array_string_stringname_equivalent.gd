# https://github.com/godotengine/godot/issues/63965

func test():
	var array_str: Array = []
	array_str.push_back("godot")
	print("StringName in Array: ", &"godot" in array_str)

	var array_sname: Array = []
	array_sname.push_back(&"godot")
	print("String in Array: ", "godot" in array_sname)

	# Not equal because the values are different types.
	print("Arrays not equal: ", array_str != array_sname)

	var string_array: Array[String] = []
	var stringname_array: Array[StringName] = []

	string_array.push_back(&"abc")
	print("Array[String] insert converted: ", typeof(string_array[0]) == TYPE_STRING)

	stringname_array.push_back("abc")
	print("Array[StringName] insert converted: ", typeof(stringname_array[0]) == TYPE_STRING_NAME)

	print("StringName in Array[String]: ", &"abc" in string_array)
	print("String in Array[StringName]: ", "abc" in stringname_array)

	var packed_string_array: PackedStringArray = []
	assert(!packed_string_array.push_back("abc"))
	print("StringName in PackedStringArray: ", &"abc" in packed_string_array)

	string_array.push_back("abc")
	print("StringName finds String in Array: ", string_array.find(&"abc"))

	stringname_array.push_back(&"abc")
	print("String finds StringName in Array: ", stringname_array.find("abc"))
