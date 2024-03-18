func test():
	var a = 0
	match a:
		0 when a = 1:
			print("assignment not allowed on pattern guard")
