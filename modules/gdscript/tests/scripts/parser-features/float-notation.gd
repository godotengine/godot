func test():
	# The following floating-point notations are all valid:
	assert(is_equal_approx(123., 123))
	assert(is_equal_approx(.123, 0.123))
	assert(is_equal_approx(.123e4, 1230))
	assert(is_equal_approx(123.e4, 1.23e6))
	assert(is_equal_approx(.123e-1, 0.0123))
	assert(is_equal_approx(123.e-1, 12.3))

	# Same as above, but with negative numbers.
	assert(is_equal_approx(-123., -123))
	assert(is_equal_approx(-.123, -0.123))
	assert(is_equal_approx(-.123e4, -1230))
	assert(is_equal_approx(-123.e4, -1.23e6))
	assert(is_equal_approx(-.123e-1, -0.0123))
	assert(is_equal_approx(-123.e-1, -12.3))

	# Same as above, but with explicit positive numbers (which is redundant).
	assert(is_equal_approx(+123., +123))
	assert(is_equal_approx(+.123, +0.123))
	assert(is_equal_approx(+.123e4, +1230))
	assert(is_equal_approx(+123.e4, +1.23e6))
	assert(is_equal_approx(+.123e-1, +0.0123))
	assert(is_equal_approx(+123.e-1, +12.3))
