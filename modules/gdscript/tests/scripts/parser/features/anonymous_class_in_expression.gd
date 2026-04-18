class Base:
	func make() -> int:
		return 0


func take(obj: Base) -> int:
	return obj.make()


func test():
	# Parenthesized anonymous class in a larger expression.
	var a: int = (Base.new(): func make() -> int: return 1).make()
	print(a)

	# Anonymous class as a call argument.
	var b = take(Base.new(): func make() -> int: return 2)
	print(b)

	# Anonymous class inside an `if` condition via parentheses.
	if (Base.new(): func make() -> int: return 3).make() == 3:
		print("if ok")

	# Anonymous class inside an array literal.
	var arr: Array[Base] = [Base.new(): func make() -> int: return 4]
	print(arr[0].make())
