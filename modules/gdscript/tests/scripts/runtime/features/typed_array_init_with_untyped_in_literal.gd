func test():
	var untyped: Variant = 32
	var typed: Array[int] = [untyped]
	Utils.check(typed.get_typed_builtin() == TYPE_INT)
	Utils.check(str(typed) == '[32]')
	print('ok')
