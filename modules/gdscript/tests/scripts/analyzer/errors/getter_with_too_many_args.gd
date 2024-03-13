var a: int: get = get_named

func test():
	pass

func get_named(name: int, oops: int) -> int:
	prints("get_named", name)
	return a
