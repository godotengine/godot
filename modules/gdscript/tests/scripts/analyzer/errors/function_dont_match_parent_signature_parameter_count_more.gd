func test():
	print("Shouldn't reach this")

class Parent:
	func my_function(_par1: int) -> int:
		return 0

class Child extends Parent:
	func my_function(_pary1: int, _par2: int) -> int:
		return 0
