func test():
	# The ternary operator below returns values of different types and the
	# result is assigned to a typed variable. This will cause a run-time error
	# if the branch with the incompatible type is picked. Here, it won't happen
	# since the `false` condition never evaluates to `true`. Instead, a warning
	# will be emitted.
	var __: int = 25
	__ = "hello" if false else -2
