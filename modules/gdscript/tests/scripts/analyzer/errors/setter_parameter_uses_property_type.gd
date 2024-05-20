var with_setter := 0:
	set(val):
		var x: String = val
		with_setter = val

func test():
	with_setter = 1
	print(with_setter)
