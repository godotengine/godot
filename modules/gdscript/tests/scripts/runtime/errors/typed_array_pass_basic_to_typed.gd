func expect_typed(typed: Array[int]):
	print(typed.size())

func test():
	var basic := [1]
	expect_typed(basic)
	print('not ok')
