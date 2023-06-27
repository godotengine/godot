func coroutine() -> int:
	@warning_ignore("redundant_await")
	await 0
	return 1

func not_coroutine() -> int:
	return 2

func test():
	var a := await coroutine()
	@warning_ignore("redundant_await")
	var b := await not_coroutine()
	@warning_ignore("redundant_await")
	var c := await 3
	prints(a, b, c)
