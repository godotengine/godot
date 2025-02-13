# https://github.com/godotengine/godot/issues/71172

func test():
	@warning_ignore("narrowing_conversion")
	var foo: int = 0.0
	print(typeof(foo) == TYPE_INT)
	var dict: Dictionary = {"a": 0.0}
	foo = dict.get("a")
	print(typeof(foo) == TYPE_INT)
