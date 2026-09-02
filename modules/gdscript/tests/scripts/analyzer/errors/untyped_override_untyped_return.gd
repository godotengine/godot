@abstract class A:
	@abstract func abstract_untyped_1()
	@abstract func abstract_untyped_2()
	@abstract func abstract_untyped_3()
	@abstract func abstract_untyped_4()

	@abstract func abstract_void_1() -> void
	@abstract func abstract_void_2() -> void
	@abstract func abstract_void_3() -> void
	@abstract func abstract_void_4() -> void

	@abstract func abstract_variant_1() -> Variant
	@abstract func abstract_variant_2() -> Variant
	@abstract func abstract_variant_3() -> Variant
	@abstract func abstract_variant_4() -> Variant

	@abstract func abstract_int_1() -> int
	@abstract func abstract_int_2() -> int
	@abstract func abstract_int_3() -> int
	@abstract func abstract_int_4() -> int

	func concrete_untyped_pass_1(): pass
	func concrete_untyped_pass_2(): pass
	func concrete_untyped_pass_3(): pass
	func concrete_untyped_pass_4(): pass

	func concrete_untyped_return_empty_1(): return
	func concrete_untyped_return_empty_2(): return
	func concrete_untyped_return_empty_3(): return
	func concrete_untyped_return_empty_4(): return

	func concrete_untyped_return_null_1(): return null
	func concrete_untyped_return_null_2(): return null
	func concrete_untyped_return_null_3(): return null
	func concrete_untyped_return_null_4(): return null

	func concrete_untyped_return_int_1(): return 0
	func concrete_untyped_return_int_2(): return 0
	func concrete_untyped_return_int_3(): return 0
	func concrete_untyped_return_int_4(): return 0

	func concrete_void_1() -> void: pass
	func concrete_void_2() -> void: pass
	func concrete_void_3() -> void: pass
	func concrete_void_4() -> void: pass

	func concrete_variant_1() -> Variant: return null
	func concrete_variant_2() -> Variant: return null
	func concrete_variant_3() -> Variant: return null
	func concrete_variant_4() -> Variant: return null

	func concrete_int_1() -> int: return 0
	func concrete_int_2() -> int: return 0
	func concrete_int_3() -> int: return 0
	func concrete_int_4() -> int: return 0

class B extends A:
	func abstract_untyped_1(): pass
	func abstract_untyped_2(): return
	func abstract_untyped_3(): return null
	func abstract_untyped_4(): return 0

	func abstract_void_1(): pass
	func abstract_void_2(): return
	func abstract_void_3(): return null # Error.
	func abstract_void_4(): return 0 # Error.

	func abstract_variant_1(): pass # Error.
	func abstract_variant_2(): return # TODO: Treated as `return null`, but should produce an empty return error.
	func abstract_variant_3(): return null
	func abstract_variant_4(): return 0

	func abstract_int_1(): pass # Error.
	func abstract_int_2(): return # Error. # TODO: Treated as `return null`. It would be nice to clarify the error.
	func abstract_int_3(): return null # Error.
	func abstract_int_4(): return 0

	# `concrete_untyped_*()` overrides should not produce errors.

	func concrete_untyped_pass_1(): pass
	func concrete_untyped_pass_2(): return
	func concrete_untyped_pass_3(): return null
	func concrete_untyped_pass_4(): return 0

	func concrete_untyped_return_empty_1(): pass
	func concrete_untyped_return_empty_2(): return
	func concrete_untyped_return_empty_3(): return null
	func concrete_untyped_return_empty_4(): return 0

	func concrete_untyped_return_null_1(): pass
	func concrete_untyped_return_null_2(): return
	func concrete_untyped_return_null_3(): return null
	func concrete_untyped_return_null_4(): return 0

	func concrete_untyped_return_int_1(): pass
	func concrete_untyped_return_int_2(): return
	func concrete_untyped_return_int_3(): return null
	func concrete_untyped_return_int_4(): return 0

	# Same as for abstract methods.

	func concrete_void_1(): pass
	func concrete_void_2(): return
	func concrete_void_3(): return null # Error.
	func concrete_void_4(): return 0 # Error.

	func concrete_variant_1(): pass # Error.
	func concrete_variant_2(): return # TODO: Treated as `return null`, but should produce an empty return error.
	func concrete_variant_3(): return null
	func concrete_variant_4(): return 0

	func concrete_int_1(): pass # Error.
	func concrete_int_2(): return # Error. # TODO: Treated as `return null`. It would be nice to clarify the error.
	func concrete_int_3(): return null # Error.
	func concrete_int_4(): return 0

func test():
	pass
