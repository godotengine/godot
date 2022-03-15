func test():
	print("Shouldn't reach this")

class Parent:
	func my_function() -> int:
		return 0

class Child extends Parent:
	func my_function() -> Vector2:
		return Vector2()
