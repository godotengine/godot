class A:
	func variant_to_int() -> Variant: return 0
	func variant_to_node() -> Variant: return null
	func node_to_node_2d() -> Node: return null

	func untyped_to_void(): pass
	func untyped_to_variant(): pass
	func untyped_to_int(): pass
	func untyped_to_node(): pass

	func void_to_untyped() -> void: pass
	func variant_to_untyped() -> Variant: return null
	func int_to_untyped() -> int: return 0
	func node_to_untyped() -> Node: return null

class B extends A:
	@override
	func variant_to_int() -> int: return 0
	@override
	func variant_to_node() -> Node: return null
	@override
	func node_to_node_2d() -> Node2D: return null

	@override
	func untyped_to_void() -> void: pass
	@override
	func untyped_to_variant() -> Variant: return null
	@override
	func untyped_to_int() -> int: return 0
	@override
	func untyped_to_node() -> Node: return null

	@override
	func void_to_untyped(): pass
	@override
	func variant_to_untyped(): pass
	@override
	func int_to_untyped(): pass
	@override
	func node_to_untyped(): pass

func test():
	pass
