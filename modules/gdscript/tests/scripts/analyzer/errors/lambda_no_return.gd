func test():
	var lambda := func() -> int:
		print('no return')
	lambda.call()
