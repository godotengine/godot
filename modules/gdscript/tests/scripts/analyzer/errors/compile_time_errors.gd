func test():
	# GH-89357
	print(Color(""))
	print(Color("invalid"))

	# GH-120049
	print(char(-1))
	print(char(0)) # The NUL character is not currently supported.
	print(char(0xD800))
	print(char(0x110000))
