func test():
	# All combinations of 1/2/3 arguments, each being int/float.

	for number in range(5):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(5.2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")


	for number in range(1, 5):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1, 5.2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1.2, 5):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1.2, 5.2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")


	for number in range(1, 5, 2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1, 5, 2.2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1, 5.2, 2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1, 5.2, 2.2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1.2, 5, 2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1.2, 5.2, 2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1.2, 5, 2.2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")

	for number in range(1.2, 5.2, 2.2):
		if typeof(number) != TYPE_INT:
			print("Number returned from `range` was not an int!")
