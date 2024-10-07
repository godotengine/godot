# GH-92217
# TODO: Add more tests.

static var static_var: int:
	set(value):
		prints("set static_var", value)
	get:
		print("get static_var")
		return 0

var member_var: int:
	set(value):
		prints("set member_var", value)
	get:
		print("get member_var")
		return 0

func test():
	var lambda := func ():
		var _tmp := static_var
		_tmp = member_var

		static_var = 1
		member_var = 1

	lambda.call()
