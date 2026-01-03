func test():
	var untyped: Variant = 32
	var typed: Dictionary[int, int] = { untyped: untyped }
	Utils.check(typed.get_typed_key_builtin() == TYPE_INT)
	Utils.check(typed.get_typed_value_builtin() == TYPE_INT)
	Utils.check(str(typed) == '{ 32: 32 }')
	print('ok')
