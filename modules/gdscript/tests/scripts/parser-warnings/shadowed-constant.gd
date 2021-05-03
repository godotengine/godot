# See also `parser-errors/redefine-class-constant.gd`.
const TEST = 25


func test():
	# Warning here. This is not an error because a new constant is created,
	# rather than attempting to set the value of an existing constant.
	const TEST = 50
