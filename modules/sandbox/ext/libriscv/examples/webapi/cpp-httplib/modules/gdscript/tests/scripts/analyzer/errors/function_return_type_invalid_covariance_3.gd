class A:
	func f() -> Node:
		return null

class B extends A:
	func f() -> void: # No `is_type_compatible()` misuse.
		return

func test():
	pass
