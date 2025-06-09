extends RefCounted

abstract class A:
	# Currently, an abstract function cannot be static.
	abstract static func f()

func test():
	pass
