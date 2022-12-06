# https://github.com/godotengine/godot/issues/64171

func test():
	print("Compare ==: ", "abc" == &"abc")
	print("Compare ==: ", &"abc" == "abc")
	print("Compare !=: ", "abc" != &"abc")
	print("Compare !=: ", &"abc" != "abc")

	print("Concat: ", "abc" + &"def")
	print("Concat: ", &"abc" + "def")
	print("Concat: ", &"abc" + &"def")
