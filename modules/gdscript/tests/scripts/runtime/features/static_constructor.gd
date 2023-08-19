@static_unload

static var foo = "bar"

static func _static_init():
	print("static init called")
	prints("foo is", foo)

func test():
	var _lambda = func _static_init():
		print("lambda does not conflict with static constructor")

	print("done")
