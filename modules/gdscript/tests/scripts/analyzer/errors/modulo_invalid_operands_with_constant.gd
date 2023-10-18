# https://github.com/godotengine/godot/issues/80486

func test1():
	var result = randi() % floor(10.0)
	print("Result: ", result)

func test2():
	var a: int = 1
	const b: Variant = 10.0
	var result = a % b
	print("Result: ", result)

func test():
	test1()
	test2()
