# https://github.com/godotengine/godot/issues/50894

func test():
	print(await not_coroutine())


func not_coroutine() -> String:
	return "awaited"
