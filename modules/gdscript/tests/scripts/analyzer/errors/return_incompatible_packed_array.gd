func get_string_array() -> Array[String]:
	var result := PackedInt32Array()
	return result

func test():
	print(var_to_str(get_string_array()))
