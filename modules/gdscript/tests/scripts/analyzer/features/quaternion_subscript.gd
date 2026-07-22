# https://github.com/godotengine/godot/issues/120318
func test():
	var quaternion := Quaternion(1.0, 2.0, 3.0, 4.0)

	# Constant integer index.
	print(quaternion[0])
	print(quaternion[3])

	# Variable integer index.
	var index := 1
	print(quaternion[index])

	# String index still works.
	print(quaternion["z"])
