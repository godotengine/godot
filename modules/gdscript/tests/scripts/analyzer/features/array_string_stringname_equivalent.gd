func test():
	# Converted to String when initialized
	var string_array: Array[String] = [&"abc"]
	print(string_array)

	# Converted to StringName when initialized
	var stringname_array: Array[StringName] = ["abc"]
	print(stringname_array)
