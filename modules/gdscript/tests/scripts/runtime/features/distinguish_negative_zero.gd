func test():
	var zero = 0.0
	var nzero = -0.0
	assert(not is_same(zero, nzero))
	assert(not is_same(0.0, -0.0))
	print("%.1f" % zero)
	print("%.1f" % nzero)
