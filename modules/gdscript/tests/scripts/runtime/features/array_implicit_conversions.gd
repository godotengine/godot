func split_to_strings(string: String) -> Array[String]:
	var result: PackedStringArray = string.split(",")
	return result

func split_to_string_names(string: String) -> Array[StringName]:
	var result: PackedStringArray = string.split(",")
	return result

func split_to_ints(string: String) -> Array[int]:
	var result: PackedFloat64Array = string.split_floats(",")
	return result

func split_to_floats(string: String) -> Array[float]:
	var result: PackedFloat64Array = string.split_floats(",")
	return result

func test():
	# GH-114299
	var a1: Array[PackedInt32Array] = [[1]]
	var a2 = [[2]] as Array[PackedInt32Array]
	print(var_to_str(a1))
	print(var_to_str(a2))

	print(var_to_str(split_to_strings("a,b,c")))
	print(var_to_str(split_to_string_names("a,b,c")))
	print(var_to_str(split_to_ints("1,2,3")))
	print(var_to_str(split_to_floats("1,2,3")))
