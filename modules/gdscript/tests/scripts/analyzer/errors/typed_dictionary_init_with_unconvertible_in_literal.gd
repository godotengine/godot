func test():
	var unconvertible := 1
	var typed: Dictionary[Object, Object] = { unconvertible: unconvertible }
	print('not ok')
