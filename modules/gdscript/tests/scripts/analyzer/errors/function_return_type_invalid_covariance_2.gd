class A:
	func f() -> Node:
		return null

class B extends A:
	func f() -> Variant: # No `is_type_compatible()` misuse.
		return null

func test():
	pass
