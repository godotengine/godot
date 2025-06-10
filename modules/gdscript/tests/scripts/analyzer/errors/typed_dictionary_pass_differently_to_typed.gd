func expect_typed(typed: Dictionary[int, int]):
	print(typed.size())

func test():
	var differently: Dictionary[float, float] = { 1.0: 0.0 }
	expect_typed(differently)
	print('not ok')
