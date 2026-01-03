func expect_typed(typed: Dictionary[int, int]):
	print(typed.size())

func test():
	var differently: Variant = { 1.0: 0.0 } as Dictionary[float, float]
	expect_typed(differently)
	print('not ok')
