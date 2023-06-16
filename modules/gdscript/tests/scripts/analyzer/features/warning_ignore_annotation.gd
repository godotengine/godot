@warning_ignore("unused_private_class_variable")
var _unused = 2

@warning_ignore("unused_variable")
var value:
	@warning_ignore("unused_variable")
	@warning_ignore("unassigned_variable")
	set(new_value):
		var unused = 3
		var unassigned
		print(unassigned)
		value=new_value


	@warning_ignore("unused_variable")
	@warning_ignore("unassigned_variable")
	get:
		var unused = 3
		var unassigned
		print(unassigned)
		return value


@warning_ignore("unused_variable")
func test():
	print("test")
	var unused = 3

	@warning_ignore("redundant_await")
	print(await regular_func())

	print("done")

func regular_func() -> int:
	return 0
