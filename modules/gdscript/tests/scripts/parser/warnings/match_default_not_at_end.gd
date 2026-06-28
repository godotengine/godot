func test():
	var i = 25
	# The default branch (`_`) should be at the end of the `match` statement.
	# Otherwise, a warning will be emitted
	match i:
		_:
			print("default")
		25:
			print("is 25")
