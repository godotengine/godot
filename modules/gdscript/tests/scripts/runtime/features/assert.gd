func get_condition(x: bool) -> bool:
	print("get_condition(%s)" % x)
	return x

func get_message(x: bool) -> String:
	print("get_message(%s)" % x)
	return "message"

func test():
	assert(get_condition(false), get_message(false))
