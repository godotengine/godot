func test():
	# The following floating-point notations are all valid:
	print(is_equal_approx(123., 123))
	print(is_equal_approx(.123, 0.123))
	print(is_equal_approx(.123e4, 1230))
	print(is_equal_approx(123.e4, 1.23e6))
	print(is_equal_approx(.123e-1, 0.0123))
	print(is_equal_approx(123.e-1, 12.3))

	# Same as above, but with negative numbers.
	print(is_equal_approx(-123., -123))
	print(is_equal_approx(-.123, -0.123))
	print(is_equal_approx(-.123e4, -1230))
	print(is_equal_approx(-123.e4, -1.23e6))
	print(is_equal_approx(-.123e-1, -0.0123))
	print(is_equal_approx(-123.e-1, -12.3))

	# Same as above, but with explicit positive numbers (which is redundant).
	print(is_equal_approx(+123., +123))
	print(is_equal_approx(+.123, +0.123))
	print(is_equal_approx(+.123e4, +1230))
	print(is_equal_approx(+123.e4, +1.23e6))
	print(is_equal_approx(+.123e-1, +0.0123))
	print(is_equal_approx(+123.e-1, +12.3))
