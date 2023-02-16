# https://github.com/godotengine/godot/issues/54944

var x = 1

func test():
	print(x)
	@warning_ignore("unused_variable", "shadowed_variable")
	var x = 2
