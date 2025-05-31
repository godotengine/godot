class A:
	@virtual func int_to_variant(_p: int): pass
	@virtual func node_to_variant(_p: Node): pass
	@virtual func node_2d_to_node(_p: Node2D): pass

	@virtual func variant_to_untyped(_p: Variant): pass
	@virtual func int_to_untyped(_p: int): pass
	@virtual func node_to_untyped(_p: Node): pass

class B extends A:
	@override func int_to_variant(_p: Variant): pass
	@override func node_to_variant(_p: Variant): pass
	@override func node_2d_to_node(_p: Node): pass

	@override func variant_to_untyped(_p): pass
	@override func int_to_untyped(_p): pass
	@override func node_to_untyped(_p): pass

func test():
	pass
