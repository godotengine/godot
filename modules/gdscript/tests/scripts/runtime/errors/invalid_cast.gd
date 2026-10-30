func subtest_wrong_builtin():
	var integer: Variant = 1
	print(integer as Array)

func subtest_builtin_as_object():
	var integer: Variant = 1
	print(integer as Node)

func subtest_object_as_builtin():
	var object: Variant = RefCounted.new()
	print(object as int)

func subtest_freed_object():
	var node := Node.new()
	node.free()
	print(node as Node2D)

func subtest_wrong_typed_array():
	var string_array: Array[String] = []
	var array: Array = string_array
	print(array as Array[RID])

func subtest_wrong_typed_dict():
	var string_dict: Dictionary[String, String] = {}
	var dict: Dictionary = string_dict
	(func subtest_wrong_typed_dict(): print(dict as Dictionary[RID, RID])).call()
	(func subtest_wrong_typed_dict(): print(dict as Dictionary[String, Variant])).call()
	(func subtest_wrong_typed_dict(): print(dict as Dictionary[Variant, String])).call()
	var string_dict2: Dictionary[String, Variant] = {}
	dict = string_dict2
	(func subtest_wrong_typed_dict(): print(dict as Dictionary[Variant, String])).call()
	(func subtest_wrong_typed_dict(): print(dict as Dictionary[String, String])).call()
	var string_dict3: Dictionary[Variant, String] = {}
	dict = string_dict3
	(func subtest_wrong_typed_dict(): print(dict as Dictionary[String, Variant])).call()
	(func subtest_wrong_typed_dict(): print(dict as Dictionary[String, String])).call()

func test():
	subtest_wrong_builtin()
	subtest_builtin_as_object()
	subtest_object_as_builtin()
	subtest_freed_object()
	subtest_wrong_typed_array()
	subtest_wrong_typed_dict()
