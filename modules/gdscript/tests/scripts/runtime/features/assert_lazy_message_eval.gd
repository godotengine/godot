# GH-78610

func get_condition(x: bool) -> bool:
	print("get_condition(%s)" % x)
	return x

func get_message(x: bool) -> String:
	print("get_message(%s)" % x)
	return "message"

func test():
	# Condition is true, so message is not evaluated.
	assert(get_condition(true), get_message(true))
