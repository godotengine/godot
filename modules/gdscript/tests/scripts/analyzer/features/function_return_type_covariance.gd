class A:
	@virtual func variant_to_int() -> Variant: return 0
	@virtual func variant_to_node() -> Variant: return null
	@virtual func node_to_node_2d() -> Node: return null

	@virtual func untyped_to_void(): pass
	@virtual func untyped_to_variant(): pass
	@virtual func untyped_to_int(): pass
	@virtual func untyped_to_node(): pass

	@virtual func void_to_untyped() -> void: pass
	@virtual func variant_to_untyped() -> Variant: return null
	@virtual func int_to_untyped() -> int: return 0
	@virtual func node_to_untyped() -> Node: return null

class B extends A:
	@override func variant_to_int() -> int: return 0
	@override func variant_to_node() -> Node: return null
	@override func node_to_node_2d() -> Node2D: return null

	@override func untyped_to_void() -> void: pass
	@override func untyped_to_variant() -> Variant: return null
	@override func untyped_to_int() -> int: return 0
	@override func untyped_to_node() -> Node: return null

	@override func void_to_untyped(): pass
	@override func variant_to_untyped(): pass
	@override func int_to_untyped(): pass
	@override func node_to_untyped(): pass

func test():
	pass
