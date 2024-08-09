class A:
	func f(x: int, ...args: Array) -> void:
		prints(x, args)

class B extends A:
	func f(x: int) -> void:
		print(x)

func test():
	pass
