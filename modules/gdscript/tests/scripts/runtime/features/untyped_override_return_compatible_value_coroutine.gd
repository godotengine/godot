# GH-121584

class A:
	signal some_signal()

	func bool_coroutine() -> bool:
		await some_signal
		return true

	func resource_coroutine() -> Resource:
		await some_signal
		return null

class B extends A:
	func untyped_func_bool():
		return true

	func untyped_func_resource():
		return Resource.new()

	func bool_coroutine():
		@warning_ignore("redundant_await")
		return await untyped_func_bool()

	func resource_coroutine():
		@warning_ignore("redundant_await")
		return await untyped_func_resource()

func test():
	var b := B.new()

	var result_bool: bool = await b.bool_coroutine()
	print(var_to_str(result_bool))

	var result_resource: Resource = await b.resource_coroutine()
	print(result_resource.get_class())
