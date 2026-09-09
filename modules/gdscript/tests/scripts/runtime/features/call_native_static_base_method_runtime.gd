class_name TestCallNativeStatic
extends JSON

func test():
	var s: GDScript = get_script()
	@warning_ignore("unsafe_method_access")
	print(s.stringify("test"))
	print(s.call(&"stringify", "test"))
	print(TestCallNativeStatic.stringify("test"))
