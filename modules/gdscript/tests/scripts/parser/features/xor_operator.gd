func test():
	print("---")
	print(0 xor 0)
	print(0 xor 1)
	print(1 xor 0)
	print(1 xor 1)

	print("---")
	print(1 xor 1 or 1)
	print((1 xor 1) or 1)
	print(1 xor (1 or 1))

	print("---")
	print(1 xor 0 and 0)
	print((1 xor 0) and 0)
	print(1 xor (0 and 0))
