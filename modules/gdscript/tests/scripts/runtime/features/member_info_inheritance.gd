# GH-82169

@warning_ignore_start("unused_signal")

@abstract class A:
	@abstract func test_abstract_func_1()
	@abstract func test_abstract_func_2()
	func test_override_func_1(): pass
	func test_override_func_2(): pass

class B extends A:
	static var test_static_var_b1
	static var test_static_var_b2
	var test_var_b1
	var test_var_b2
	static func test_static_func_b1(): pass
	static func test_static_func_b2(): pass
	func test_abstract_func_1(): pass
	func test_abstract_func_2(): pass
	func test_override_func_1(): pass
	func test_override_func_2(): pass
	func test_func_b1(): pass
	func test_func_b2(): pass
	signal test_signal_b1()
	signal test_signal_b2()

class C extends B:
	static var test_static_var_c1
	static var test_static_var_c2
	var test_var_c1
	var test_var_c2
	static func test_static_func_c1(): pass
	static func test_static_func_c2(): pass
	func test_abstract_func_1(): pass
	func test_abstract_func_2(): pass
	func test_override_func_1(): pass
	func test_override_func_2(): pass
	func test_func_c1(): pass
	func test_func_c2(): pass
	signal test_signal_c1()
	signal test_signal_c2()

func test_property_signature(name: String, base: Object, is_static: bool = false) -> void:
	prints("---", name, "---")
	for property in base.get_property_list():
		if str(property.name).begins_with("test_"):
			print(Utils.get_property_signature(property, null, is_static))

func test_method_signature(name: String, base: Object) -> void:
	prints("---", name, "---")
	for method in base.get_method_list():
		if str(method.name).begins_with("test_"):
			print(Utils.get_method_signature(method))

func test_signal_signature(name: String, base: Object) -> void:
	prints("---", name, "---")
	for method in base.get_signal_list():
		if str(method.name).begins_with("test_"):
			print(Utils.get_method_signature(method, true))

func test():
	var b := B.new()
	var c := C.new()

	print("=== Class Properties ===")
	test_property_signature("A", A as GDScript, true)
	test_property_signature("B", B as GDScript, true)
	test_property_signature("C", C as GDScript, true)
	print("=== Member Properties ===")
	test_property_signature("B", b)
	test_property_signature("C", c)
	print("=== Class Methods ===")
	test_method_signature("A", A as GDScript)
	test_method_signature("B", B as GDScript)
	test_method_signature("C", C as GDScript)
	print("=== Member Methods ===")
	test_method_signature("B", b)
	test_method_signature("C", c)
	print("=== Signals ===")
	test_signal_signature("B", b)
	test_signal_signature("C", c)
