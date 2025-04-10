func test():
	var lambda := func() -> int:
		return 'string'
	print(lambda.call())
