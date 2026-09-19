class Inner:
	var prop = "Inner"

class SubOfInner extends Inner:
	pass

class SubOfGloballyNamed extends TypedDictionaryWithCustomClassBase:
	pass

class SubOfNative extends Node2D:
	pass

var dict: Dictionary[int, Inner] = { 0: Inner.new() }


func test_subclass_key() -> void:
	var inner_key_dict: Dictionary[Inner, String] = {}
	var inner := Inner.new()
	var sub_of_inner := SubOfInner.new()
	inner_key_dict[inner] = "inner"
	inner_key_dict[sub_of_inner] = "sub of inner"
	print(inner_key_dict[inner])
	print(inner_key_dict[sub_of_inner])
	print("test_subclass_key: ok")


func test_multi_file() -> void:
	var globally_named_key_dict: Dictionary[TypedDictionaryWithCustomClassBase, String] = {}
	var sub := SubOfGloballyNamed.new()
	globally_named_key_dict[sub] = "subclass instance"
	var as_base: TypedDictionaryWithCustomClassBase = SubOfGloballyNamed.new()
	globally_named_key_dict[as_base] = "base-typed variable"
	print(globally_named_key_dict[sub])
	print(globally_named_key_dict[as_base])
	print("test_multi_file: ok")


func test_native() -> void:
	var node_key_dict: Dictionary[Node, String] = {}
	var node_2d := Node2D.new()
	var node_3d := Node3D.new()
	var sub_of_native := SubOfNative.new()
	node_key_dict[node_2d] = "node2d"
	node_key_dict[node_3d] = "node3d"
	node_key_dict[sub_of_native] = "sub of native"
	print(node_key_dict[node_2d])
	print(node_key_dict[node_3d])
	print(node_key_dict[sub_of_native])
	node_2d.free()
	node_3d.free()
	sub_of_native.free()
	print("test_native: ok")


func test_class_reference_key() -> void:
	# A script class reference is itself an `Object` instance.
	var object_key_dict: Dictionary[Object, String] = {}
	object_key_dict[Inner] = "class reference key"
	print(object_key_dict[Inner])
	print("test_class_reference_key: ok")


func test():
	var element: Inner = dict[0]
	print(element.prop)

	test_subclass_key()
	test_multi_file()
	test_native()
	test_class_reference_key()
