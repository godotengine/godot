# https://github.com/godotengine/godot/issues/93952

func foo():
	pass

func test():
	var a: int

	var lambda_self := func (x: int) -> void:
		foo()
		print(a, x)

	print(lambda_self.get_argument_count())  # Should print 1.

	var lambda_non_self := func (x: int) -> void:
		print(a, x)

	print(lambda_non_self.get_argument_count())  # Should print 1.
