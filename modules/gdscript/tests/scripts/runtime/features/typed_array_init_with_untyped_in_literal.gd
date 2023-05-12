func test():
	var untyped: Variant = 32
	var typed: Array[int] = [untyped]
	assert(typed.get_typed_builtin() == TYPE_INT)
	assert(str(typed) == '[32]')
	print('ok')
