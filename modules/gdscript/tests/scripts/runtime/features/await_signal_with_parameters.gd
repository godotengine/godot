signal no_parameters()
signal one_parameter(number)
signal two_parameters(number1, number2)

func await_no_parameters():
	var result = await no_parameters
	print(result)

func await_one_parameter():
	var result = await one_parameter
	print(result)

func await_two_parameters():
	var result = await two_parameters
	print(result)

func test():
	await_no_parameters()
	no_parameters.emit()

	await_one_parameter()
	one_parameter.emit(1)

	await_two_parameters()
	two_parameters.emit(1, 2)
