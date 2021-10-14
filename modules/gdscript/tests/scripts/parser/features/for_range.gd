func test():
	for i in range(5):
		print(i)

	print()

	# Equivalent to the above `for` loop:
	for i in 5:
		print(i)

	print()

	for i in range(1, 5):
		print(i)

	print()

	for i in range(1, -5, -1):
		print(i)

	print()

	for i in [2, 4, 6, -8]:
		print(i)

	print()

	for i in [true, false]:
		print(i)

	print()

	for i in [Vector2i(10, 20), Vector2i(-30, -40)]:
		print(i)

	print()

	for i in "Hello_Unicôde_world!_🦄":
		print(i)
