class A:
	func return_void() -> void: pass
	func return_int() -> int: return 123
	func return_node(_resource: Resource) -> Node: return null
	func return_variant() -> Variant: return null

class B extends A:
	func return_void(): return 1
	func return_int(): return "abc"
	func return_node(resource: Resource): return resource
	func return_variant(): pass

func test():
	pass
