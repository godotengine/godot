func test():
	const pa: PackedByteArray = [1, 2, 3]
	const pb: PackedByteArray = [1, 2, 3]
	assert(not is_same(pa, pb))
	assert(pa == pb)
	# Delete the part below if const packed arrays become read-only
	var _t = pa.append(4)
	assert(pa != pb)
	print(pa)
	print(pb)
