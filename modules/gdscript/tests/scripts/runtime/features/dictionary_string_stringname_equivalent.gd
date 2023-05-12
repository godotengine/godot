# https://github.com/godotengine/godot/issues/62957

func test():
	var string_dict = {}
	string_dict["abc"] = 42
	var stringname_dict = {}
	stringname_dict[&"abc"] = 24

	print("String key is TYPE_STRING: ", typeof(string_dict.keys()[0]) == TYPE_STRING)
	print("StringName key is TYPE_STRING: ", typeof(stringname_dict.keys()[0]) == TYPE_STRING)

	print("StringName gets String: ", string_dict.get(&"abc"))
	print("String gets StringName: ", stringname_dict.get("abc"))

	stringname_dict[&"abc"] = 42
	# They compare equal because StringName keys are converted to String.
	print("String Dictionary == StringName Dictionary: ", string_dict == stringname_dict)
