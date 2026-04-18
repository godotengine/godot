class Base:
	func describe() -> String:
		return "base"

	func make() -> int:
		return 0


func take(obj: Base) -> int:
	return obj.make()


func test():
	# Inline single-line body.
	var inline = Base.new(): func describe() -> String: return "inline"
	print(inline.describe())

	# Parenthesized anonymous class in a larger expression.
	var p_int: int = (Base.new(): func make() -> int: return 1).make()
	print(p_int)

	# Anonymous class as a call argument.
	var arg = take(Base.new(): func make() -> int: return 2)
	print(arg)

	# Anonymous class inside an `if` condition via parentheses.
	if (Base.new(): func make() -> int: return 3).make() == 3:
		print("if ok")

	# Anonymous class inside an array literal.
	var arr: Array[Base] = [Base.new(): func make() -> int: return 4]
	print(arr[0].make())

	# Semicolon terminates an inline func body.
	var semi = Base.new(): func make() -> int: return 5;
	print(semi.make())

	# Multiple statements in an inline body separated by semicolons.
	var multi = Base.new(): func make() -> int: var x = 10; return x + 5
	print(multi.make())
