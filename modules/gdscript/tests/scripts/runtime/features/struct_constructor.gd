struct Point:
	var x: int
	var y: int

func test():
	var point_one: Point = Point(5, 10)
	var point_two: Point = Point(7) # Second arg should fall back to 0.

	print(point_one.x)
	print(point_one.y)
	print(point_two.x)
	print(point_two.y)
