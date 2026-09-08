@warning_ignore_start("untyped_declaration")

class SetGetCoroutine:
	signal sig

	func coroutine_getter():
		await sig
		return inline_coroutine1

	func coroutine_setter(val):
		await sig
		inline_coroutine1 = val


	var inline_coroutine1:
		set(val):
			await sig
			inline_coroutine1 = val

	var inline_coroutine2:
		get:
			await sig
			return inline_coroutine2

	var inline_coroutine3:
		set(val):
			await sig
			inline_coroutine3 = val
		get:
			await sig
			return inline_coroutine3

	var func_coroutine1: set = coroutine_setter
	var func_coroutine2: get = coroutine_getter
	var func_coroutine3: get = coroutine_getter, set = coroutine_setter

class SetGetGradualTyping:
	var prop

	func typed_setter(val: String): prop = val
	func untyped_setter(val): prop = val
	func typed_getter() -> String: return prop
	func untyped_getter(): return prop

	var p1: set = typed_setter
	var p2: String: set = untyped_setter
	var p3: get = typed_getter
	var p4: String: get = untyped_getter

	var p5: String: set = untyped_setter, get = untyped_getter
	var p6: set = typed_setter, get = untyped_getter

func test():
	pass
