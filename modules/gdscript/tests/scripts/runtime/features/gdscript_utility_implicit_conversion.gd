func test():
	const COLOR = Color8(255, 0.0, false)
	var false_value := false
	@warning_ignore("narrowing_conversion")
	var color = Color8(255, 0.0, false_value)
	print(var_to_str(COLOR))
	print(var_to_str(color))

	var string := "A"
	var string_name := &"A"
	print(ord(string))
	print(ord(string_name))
