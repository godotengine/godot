struct Point:
	var x: int
	var y: int

func test():
	# Test instantiation with no arguments
	var p1 = Point()
	print(p1)
	
	# Test instantiation with arguments
	var p2 = Point(10, 20)
	print(p2)
