@warning_ignore("unused_private_class_variable")
var _unused = 2

@warning_ignore("unused_variable")
func test():
	print("test")
	var unused = 3

	@warning_ignore("redundant_await")
	print(await regular_func())

	print("done")

func regular_func() -> int:
	return 0
