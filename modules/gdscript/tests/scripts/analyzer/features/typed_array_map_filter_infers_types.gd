func half(a: int) -> float:
	return float(a) / 2.0

func test():
	var source: Array[int] = [1, 2, 3]

	var filtered := source.filter(func(a): return a > 1)
	var filtered_value: int = filtered[0]
	Utils.check(filtered_value == 2)

	var mapped_lambda := source.map(func(a: int) -> float: return float(a) / 2.0)
	var mapped_lambda_value: float = mapped_lambda[0]
	Utils.check(mapped_lambda.get_typed_builtin() == TYPE_FLOAT)
	Utils.check(is_equal_approx(mapped_lambda_value, 0.5))

	var callable: Callable = half
	var mapped_named := source.map(callable)
	var mapped_named_value: float = mapped_named[0]
	Utils.check(mapped_named.get_typed_builtin() == TYPE_FLOAT)
	Utils.check(is_equal_approx(mapped_named_value, 0.5))

	print('ok')
