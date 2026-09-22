func test():
	var i = "Hello"
	match i:
		"Hello":
			print("hello")
		"Good bye":
			print("bye")
		_:
			print("default")

	var j = 25
	match j:
		26:
			print("This won't match")
		_:
			print("This will match")

	match 0:
		pass
