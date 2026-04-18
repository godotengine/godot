class Base:
	func make() -> int:
		return 0


func test():
	# Semicolon acts as a statement terminator for the inline func body.
	var a = Base.new(): func make() -> int: return 1;
	print(a.make())

	# Multiple statements inside the inline body separated by semicolons.
	var b = Base.new(): func make() -> int: var x = 10; return x + 5
	print(b.make())
