signal my_signal()

# CI cannot test async things.
func test_signals():
	await my_signal
	var t: Signal = my_signal
	await t

func coroutine() -> void:
	@warning_ignore("redundant_await")
	await 0

func not_coroutine_variant():
	pass

func not_coroutine_void() -> void:
	pass

func test():
	const CONST_NULL = null
	var var_null = null
	var var_int: int = 1
	var var_variant: Variant = 1
	var var_array: Array = [1]

	await CONST_NULL
	await var_null
	await var_int
	await var_variant
	await var_array[0]

	await coroutine
	await coroutine()
	await coroutine.call()
	await self.coroutine()
	await call(&"coroutine")

	await not_coroutine_variant
	await not_coroutine_variant()
	await self.not_coroutine_variant()
	await not_coroutine_variant.call()
	await call(&"not_coroutine_variant")

	await not_coroutine_void
	await not_coroutine_void()
	await self.not_coroutine_void()
	await not_coroutine_void.call()
	await call(&"not_coroutine_void")

	var callable: Callable = coroutine
	await callable
	await callable.call()
	await callable.get_method()
