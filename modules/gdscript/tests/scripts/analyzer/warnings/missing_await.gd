func coroutine() -> void:
	@warning_ignore("redundant_await")
	await 0

func test():
	await coroutine()
	coroutine()
