# https://github.com/godotengine/godot/issues/41064
var x = true

func test():
	var int_var: int = 0
	var dyn_var = 2

	if x:
		dyn_var = 5
	else:
		dyn_var = Node.new()

	int_var = dyn_var
	print(int_var)
