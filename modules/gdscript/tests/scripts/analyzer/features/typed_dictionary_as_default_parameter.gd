func print_untyped(dictionary = { 0: 1 }) -> void:
	print(dictionary)
	print(dictionary.get_typed_key_builtin())
	print(dictionary.get_typed_value_builtin())

func print_inferred(dictionary := { 2: 3 }) -> void:
	print(dictionary)
	print(dictionary.get_typed_key_builtin())
	print(dictionary.get_typed_value_builtin())

func print_typed(dictionary: Dictionary[int, int] = { 4: 5 }) -> void:
	print(dictionary)
	print(dictionary.get_typed_key_builtin())
	print(dictionary.get_typed_value_builtin())

func test():
	print_untyped()
	print_inferred()
	print_typed()
	print('ok')
