# GH-82169

const Utils = preload("../../utils.notest.gd")

class A:
	static var test_static_var_a1
	static var test_static_var_a2
	var test_var_a1
	var test_var_a2
	static func test_static_func_a1(): pass
	static func test_static_func_a2(): pass
	func test_func_a1(): pass
	func test_func_a2(): pass
	signal test_signal_a1()
	signal test_signal_a2()

class B extends A:
	static var test_static_var_b1
	static var test_static_var_b2
	var test_var_b1
	var test_var_b2
	static func test_static_func_b1(): pass
	static func test_static_func_b2(): pass
	func test_func_b1(): pass
	func test_func_b2(): pass
	signal test_signal_b1()
	signal test_signal_b2()

func test():
	var b := B.new()
	for property in (B as GDScript).get_property_list():
		if str(property.name).begins_with("test_"):
			print(Utils.get_property_signature(property, true))
	print("---")
	for property in b.get_property_list():
		if str(property.name).begins_with("test_"):
			print(Utils.get_property_signature(property))
	print("---")
	for method in b.get_method_list():
		if str(method.name).begins_with("test_"):
			print(Utils.get_method_signature(method))
	print("---")
	for method in b.get_signal_list():
		if str(method.name).begins_with("test_"):
			print(Utils.get_method_signature(method, true))
