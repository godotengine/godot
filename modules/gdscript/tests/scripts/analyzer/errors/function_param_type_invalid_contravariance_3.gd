class A:
	func f(_p: int):
		pass

class B extends A:
	func f(_p: float): # No implicit conversion.
		pass

func test():
	pass
