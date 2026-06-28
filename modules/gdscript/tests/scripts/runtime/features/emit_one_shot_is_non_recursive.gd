# https://github.com/godotengine/godot/issues/89439

signal my_signal

func foo():
	print("Foo")
	my_signal.emit()

func bar():
	print("Bar")

func baz():
	print("Baz")

func test():
	@warning_ignore("return_value_discarded")
	my_signal.connect(foo, CONNECT_ONE_SHOT)
	@warning_ignore("return_value_discarded")
	my_signal.connect(bar, CONNECT_ONE_SHOT)
	@warning_ignore("return_value_discarded")
	my_signal.connect(baz)
	my_signal.emit()
