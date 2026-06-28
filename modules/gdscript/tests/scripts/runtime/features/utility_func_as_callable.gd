func test():
	print(print)
	print(len)

	prints.callv([1, 2, 3])
	print(mini.call(1, 2))
	print(len.bind("abc").call())

	const ABSF = absf
	print(ABSF.call(-1.2))
