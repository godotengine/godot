class A:
	func f() -> float:
		return 0.0

class B extends A:
	func f() -> int: # No implicit conversion.
		return 0

func test():
	pass
