func expect_typed(typed: Array[int]):
	print(typed.size())

func test():
	var differently: Array[float] = [1.0]
	expect_typed(differently)
	print('not ok')
