class Inner:
	extends RefCounted

func half(a: int) -> float:
	return float(a) / 2.0

func test():
	var source: Array[int] = [1, 2, 3]

	# filter preserves source element type.
	var filtered := source.filter(func(a): return a > 1)
	var filtered_value: int = filtered[0]
	Utils.check(filtered_value == 2)

	# map with lambda infers builtin return type.
	var mapped_lambda := source.map(func(a: int) -> float: return float(a) / 2.0)
	var mapped_lambda_value: float = mapped_lambda[0]
	Utils.check(mapped_lambda.get_typed_builtin() == TYPE_FLOAT)
	Utils.check(is_equal_approx(mapped_lambda_value, 0.5))

	# map with named callable infers builtin return type.
	var callable: Callable = half
	var mapped_named := source.map(callable)
	var mapped_named_value: float = mapped_named[0]
	Utils.check(mapped_named.get_typed_builtin() == TYPE_FLOAT)
	Utils.check(is_equal_approx(mapped_named_value, 0.5))

	# map with lambda infers native class return type.
	var mapped_native := source.map(func(_a: int) -> RefCounted: return RefCounted.new())
	Utils.check(mapped_native.get_typed_builtin() == TYPE_OBJECT)
	Utils.check(mapped_native.get_typed_class_name() == &"RefCounted")

	# map with lambda infers custom script class return type.
	var mapped_script := source.map(func(_a: int) -> Inner: return Inner.new())
	Utils.check(mapped_script.get_typed_builtin() == TYPE_OBJECT)
	Utils.check(mapped_script.get_typed_script() == Inner)

	print('ok')
