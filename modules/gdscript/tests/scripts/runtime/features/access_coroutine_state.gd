# https://github.com/godotengine/godot/issues/95210
signal _signal

func foo():
	await _signal

func test():
	@warning_ignore("missing_await")
	foo()

	var awaiting_callable: Callable = _signal.get_connections()[0].callable
	if awaiting_callable.get_object() != null:
		print("ok")
