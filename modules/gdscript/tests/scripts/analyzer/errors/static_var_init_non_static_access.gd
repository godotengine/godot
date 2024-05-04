# GH-91403

@static_unload

func non_static():
	return "non static"

static var static_var = Callable(non_static)

func test():
	print("does not run")
