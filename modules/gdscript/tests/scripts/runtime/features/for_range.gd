func test():
	# GH-83293
	for x in range(1 << 31, (1 << 31) + 3):
		print(x)
	for x in range(1 << 62, (1 << 62) + 3):
		print(x)

	# GH-107392
	var n = 1.0
	for x in range(n):
		print(x)
