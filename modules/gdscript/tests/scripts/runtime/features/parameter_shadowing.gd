# https://github.com/godotengine/godot/pull/69620

var a: int = 1

func shadow_regular_assignment(a: Variant, b: Variant) -> void:
	print(a)
	print(self.a)
	a = b
	print(a)
	print(self.a)


var v := Vector2(0.0, 0.0)

func shadow_subscript_assignment(v: Vector2, x: float) -> void:
	print(v)
	print(self.v)
	v.x += x
	print(v)
	print(self.v)


func test():
	shadow_regular_assignment('a', 'b')
	shadow_subscript_assignment(Vector2(1.0, 1.0), 5.0)
