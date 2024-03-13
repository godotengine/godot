var a: int: get = getter

func test():
	pass

func getter() -> String:
	prints("getter")
	return a
