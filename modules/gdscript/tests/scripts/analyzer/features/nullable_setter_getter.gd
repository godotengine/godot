var foo: String?:
	set(value):
		print("tried setting")
		foo = value
	get:
		print("tried getting")
		return foo

func test():
	foo = null
	print("ok" if foo == null else "nok")
