@abstract class A:
	@abstract func test_untyped_1()
	@abstract func test_untyped_2()
	@abstract func test_untyped_3()
	@abstract func test_untyped_4()
	@abstract func test_void() -> void
	@abstract func test_variant() -> Variant
	@abstract func test_int() -> int

class B extends A:
	@override
	func test_untyped_1(): pass
	@override
	func test_untyped_2(): return
	@override
	func test_untyped_3(): return null
	@override
	func test_untyped_4(): return 0
	@override
	func test_void() -> void: pass
	@override
	func test_variant() -> Variant: return null
	@override
	func test_int() -> int: return 0

func test_script_method_signature(name: String, script: Script) -> void:
	prints("---", name, "---")
	for method: Dictionary in script.get_script_method_list():
		if str(method.name).begins_with("test_"):
			print(Utils.get_method_signature(method))

func test_instance_method_signature(name: String, instance: Object) -> void:
	prints("---", name, "---")
	for method in instance.get_method_list():
		if str(method.name).begins_with("test_"):
			print(Utils.get_method_signature(method))

func test():
	var b := B.new()

	print("=== Script Methods ===")
	test_script_method_signature("A", A as GDScript)
	test_script_method_signature("B", B as GDScript)
	print("=== Instance Methods ===")
	test_instance_method_signature("B", b)
