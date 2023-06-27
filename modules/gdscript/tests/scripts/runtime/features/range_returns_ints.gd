func test():
	# All combinations of 1/2/3 arguments, each being int/float.
	# Store result in variable to ensure actual array is created (avoid `for` + `range` optimization).

	var result

	result = range(5)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(5.2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")


	result = range(1, 5)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1, 5.2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1.2, 5)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1.2, 5.2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")


	result = range(1, 5, 2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1, 5, 2.2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1, 5.2, 2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1, 5.2, 2.2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1.2, 5, 2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1.2, 5.2, 2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1.2, 5, 2.2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	result = range(1.2, 5.2, 2.2)
	for number in result:
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")
