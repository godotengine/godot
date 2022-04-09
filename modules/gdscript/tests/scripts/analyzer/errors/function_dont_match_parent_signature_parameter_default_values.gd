func test():
	print("Shouldn't reach this")

class Parent:
	func my_function(_par1: int = 0) -> int:
		return 0

class Child extends Parent:
	func my_function(_par1: int) -> int:
		return 0
