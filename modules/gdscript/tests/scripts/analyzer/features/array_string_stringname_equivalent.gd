
var m_string_array: Array[String] = [&"abc"]
var m_stringname_array: Array[StringName] = ["abc"]

func test():
	print(m_string_array)
	print(m_stringname_array)

	# Converted to String when initialized
	var string_array: Array[String] = [&"abc"]
	print(string_array)

	# Converted to StringName when initialized
	var stringname_array: Array[StringName] = ["abc"]
	print(stringname_array)
