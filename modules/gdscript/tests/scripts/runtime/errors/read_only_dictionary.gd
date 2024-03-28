func test():
	var dictionary := { "a": 0 }
	dictionary.make_read_only()
	dictionary.a = 1
