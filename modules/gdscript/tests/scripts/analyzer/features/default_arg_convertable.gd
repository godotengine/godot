func check(arg: float = 3):
	return typeof(arg) == typeof(3.0)

func test():
	if check():
		print('ok')
