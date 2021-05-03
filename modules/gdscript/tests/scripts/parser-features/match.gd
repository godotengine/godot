func test():
	var i = "Hello"
	match i:
		"Hello":
			print("hello")
			# This will fall through to the default case below.
			continue
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
