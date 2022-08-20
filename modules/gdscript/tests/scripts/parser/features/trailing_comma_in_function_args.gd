# See https://github.com/godotengine/godot/issues/41066.

func f(p, ): ## <-- no errors
	print(p)

func test():
	f(0, ) ## <-- no error
