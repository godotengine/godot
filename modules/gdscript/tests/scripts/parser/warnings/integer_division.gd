func test():
	# This should emit a warning.
	var __ = 5 / 2

	# These should not emit warnings.
	__ = float(5) / 2
	__ = 5 / float(2)
	__ = 5.0 / 2
	__ = 5 / 2.0
	__ = 5.0 / 2.0
