# https://github.com/godotengine/godot/issues/71109
# https://github.com/godotengine/godot/issues/97204
# https://github.com/godotengine/godot/issues/106692
# https://github.com/godotengine/godot/issues/109319


func test():
	(
		(
			func():
				print("lambda")
		)
	).call()

	(
		(
			func():
				(
					(
						func():
							print("nested")
					)
				).call()
		)
	).call()

	var lambda_array = [
		func():
			print("array0")
		,
		func():
			print("array1")
		,
		func():
			print("array2")
		,
	]

	for lambda in lambda_array:
		@warning_ignore("unsafe_method_access")
		lambda.call()

	var lambda_dictionary = {
		0: func():
			print("dict0")
		,
		1: func():
			print("dict1")
		,
	}

	@warning_ignore("unsafe_method_access")
	lambda_dictionary[0].call()
	@warning_ignore("unsafe_method_access")
	lambda_dictionary[1].call()
