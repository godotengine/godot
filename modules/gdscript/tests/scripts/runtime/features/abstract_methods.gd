@abstract class A:
	@abstract func get_text_1() -> String
	@abstract func get_text_2() -> String

	# No `UNUSED_PARAMETER` warning.
	@abstract func func_with_param(param: int) -> int
	@abstract func func_with_rest_param(...args: Array) -> int
	@abstract func func_with_semicolon() -> int;
	@abstract func func_1() -> int; @abstract func func_2() -> int
	@abstract func func_without_return_type()

	func print_text_1() -> void:
		print(get_text_1())

@abstract class B extends A:
	func get_text_1() -> String:
		return "text_1b"

	func print_text_2() -> void:
		print(get_text_2())

class C extends B:
	func get_text_2() -> String:
		return "text_2c"

	func func_with_param(param: int) -> int: return param
	func func_with_rest_param(...args: Array) -> int: return args.size()
	func func_with_semicolon() -> int: return 0
	func func_1() -> int: return 0
	func func_2() -> int: return 0
	func func_without_return_type(): pass

@abstract class D extends C:
	@abstract func get_text_1() -> String

	func get_text_2() -> String:
		return super() + " text_2d"

class E extends D:
	func get_text_1() -> String:
		return "text_1e"

func test():
	var c := C.new()
	c.print_text_1()
	c.print_text_2()

	var e := E.new()
	e.print_text_1()
	e.print_text_2()
