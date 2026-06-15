# https://github.com/godotengine/godot/issues/120318
func test():
	var q := Quaternion(1, 2, 3, 4)
	print(q[0])
	print(q[1])
	print(q[2])
	print(q[3])
