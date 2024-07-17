# https://github.com/godotengine/godot/issues/85952

var vec: Vector2 = Vector2.ZERO:
	set(new_vec):
		prints("setting vec from", vec, "to", new_vec)
		if new_vec == Vector2(1, 1):
			vec = new_vec

func test():
	vec.x = 2
	vec.y = 2

	prints("vec is", vec)
