func test():
	var lambda_0 := func() -> void:
		print(0)
	lambda_0.call()

	var lambda_1 := func(printed: int) -> void:
		print(printed)
	lambda_1.call(1)

	var lambda_2 := func(identity: int) -> int:
		return identity
	print(lambda_2.call(2))
