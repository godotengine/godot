func test():
	var dict = { a = 1 }
	match 2:
		dict["a"]:
			print("not allowed")
