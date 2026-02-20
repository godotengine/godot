@abstract class A:
	@abstract func my_func() -> String

class B extends A:
	func my_func() -> String:
		prints("B.my_func")
		return "abc"

func return_int(_instance: Object, args: Array, previous: Callable) -> int:
	prints("return_int", args, var_to_str(previous.callv(args)))
	return 123

func test():
	print((A as GDScript).patch_method(&"my_func", return_int))
	print((B as GDScript).patch_method(&"non_existent", return_int))
	print((B as GDScript).patch_method(&"my_func", return_int))
	print(var_to_str(B.new().my_func()))
