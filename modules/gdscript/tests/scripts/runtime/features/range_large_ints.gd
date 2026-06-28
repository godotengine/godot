func test():
	# GH-109376
	# Note: unlike "for range", which is iterated in place, this helper function generates an array of all values in range.
	var result

	result = range(2147483640, 2147483647) # Range below 32-bit size limit.
	print(result)

	result = range(2147483640, 2147483647 + 1) # Range 1 over 32-bit size limit.
	print(result)

	result = range(9922147483640, 9922147483647) # Range significantly over 32-bit size limit.
	print(result)
