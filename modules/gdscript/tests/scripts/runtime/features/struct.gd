struct Point:
	var x: float
	var y: float
	var name: String = "Origin"

func test():
	var point: Point = Point()

	# Check default values.
	print(point.x)
	print(point.y)
	print(point.name)

	# Mutate values.
	point.x = 10.5
	point.y = 20.0
	point.name = "Target"

	# Check mutated values.
	print(point.x)
	print(point.y)
	print(point.name)
