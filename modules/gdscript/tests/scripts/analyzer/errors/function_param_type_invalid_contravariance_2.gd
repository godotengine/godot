class A:
	func f(_p: Variant):
		pass

class B extends A:
	func f(_p: Node): # No `is_type_compatible()` misuse.
		pass

func test():
	pass
