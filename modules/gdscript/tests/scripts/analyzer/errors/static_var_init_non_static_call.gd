@static_unload

func non_static():
	return "non static"

static var static_var = non_static()

func test():
	print("does not run")
