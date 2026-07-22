# GH-121584

class A:
	signal some_signal()

	func bool_coroutine() -> bool:
		await some_signal
		return true

class B extends A:
	func untyped_func():
		return "string"

	func bool_coroutine():
		@warning_ignore("redundant_await")
		return await untyped_func()

func test():
	var b := B.new()
	var res: bool = await b.bool_coroutine()
	print(var_to_str(res))
