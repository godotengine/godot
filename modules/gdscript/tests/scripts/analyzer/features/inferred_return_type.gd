# https://github.com/godotengine/godot/issues/61159

func get_param():
	return null

func test():
	var v = get_param()
	v = get_param()
	print(v)
