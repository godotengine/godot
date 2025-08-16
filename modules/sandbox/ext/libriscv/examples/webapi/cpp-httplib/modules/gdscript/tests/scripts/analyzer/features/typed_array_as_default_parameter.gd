func print_untyped(array = [0]) -> void:
	print(array)
	print(array.get_typed_builtin())

func print_inferred(array := [1]) -> void:
	print(array)
	print(array.get_typed_builtin())

func print_typed(array: Array[int] = [2]) -> void:
	print(array)
	print(array.get_typed_builtin())

func test():
	print_untyped()
	print_inferred()
	print_typed()
	print('ok')
