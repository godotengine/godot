extends RefCounted

abstract class A:
	# Currently, an abstract function cannot be static.
	static abstract func f()

func test():
	pass
