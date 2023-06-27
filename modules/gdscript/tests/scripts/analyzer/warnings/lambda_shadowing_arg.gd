var shadow: int

func test():
	var lambda := func(shadow: String) -> void:
		print(shadow)
	lambda.call('shadow')
