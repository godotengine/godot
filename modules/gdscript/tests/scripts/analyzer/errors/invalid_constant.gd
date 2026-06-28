func test():
	var i = 12
	# Constants must be made of a constant, deterministic expression.
	# A constant that depends on a variable's value is not a constant expression.
	const TEST = 13 + i
