func test():
	var untyped: Variant = 32
	var typed: Dictionary[int, int] = { untyped: untyped }
	assert(typed.get_typed_key_builtin() == TYPE_INT)
	assert(typed.get_typed_value_builtin() == TYPE_INT)
	assert(str(typed) == '{ 32: 32 }')
	print('ok')
