class A:
	func f(x: int, ...args: Array) -> void:
		prints(x, args)

class B extends A:
	func f(x: int, y: int, ...args: Array) -> void:
		prints(x, y, args)

func test():
	pass
