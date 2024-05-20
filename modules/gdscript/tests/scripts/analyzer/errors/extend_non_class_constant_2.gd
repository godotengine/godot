# GH-75870

class A:
	const X = 1

const Y = A.X # A.X is now resolved.

class B extends A.X:
	pass

func test():
	pass
